import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging.version import parse as V
from typeguard import check_argument_types
from torch import nn
import torch.nn.functional as F
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def pad_list(xs, pad_value, encoder_out=None):
    n_batch = len(xs)
    max_len = max(len(x) for x in xs)
    pad = encoder_out.new(n_batch, max_len).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : len(xs[i])] = torch.tensor(xs[i])

    return pad

def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_phone_dict(utt2align):
    #logging.warning(f"utt2align:{utt2align}")
    with open(utt2align, "r") as f:
        lines = f.readlines()
    """
    return: phones_dict: {uttid:{
                      phone_list:[1,5,7,...], # 1 denotes "SIL"
                      phone_ratio:[0.1,0.2,...],
                      mask_sil_list:[0,1,1,...],
                      }
                   uttid:{

                   }
                  }
    """

    phones_dict = {}

    for line in lines:
        phone_list = []
        phone_ratio = []
        total_count = 0
        phone_duration = 1

        utt_id = line.strip().split(" ")[0]
        phones_list = line.strip().split(" ")[1:]

        for i in range(1, len(phones_list)):
            if phones_list[i] == phones_list[i-1]:
                phone_duration += 1
                if i == len(phones_list) - 1:
                    phone_list.append(int(phones_list[i]))
                    phone_ratio.append(phone_duration)
            else:
                phone_list.append(int(phones_list[i-1]))
                phone_ratio.append(phone_duration)
                phone_duration = 1
                if i == len(phones_list) - 1:
                    phone_list.append(int(phones_list[i]))
                    phone_ratio.append(phone_duration)
            total_count += 1
        for i in range(0,len(phone_ratio)):
            phone_ratio[i] = phone_ratio[i] / total_count

        phones_dict[utt_id] = {"phone_list": phone_list, "phone_ratio": phone_ratio}

    return phones_dict
def get_word_len_align(word_len_align):
    word_align_dict = {}
    with open(word_len_align,"r") as f:
        lines = f.readlines()
    for line in lines:
        uttid = line.strip().split(" ")[0]
        align_list = line.strip().split(" ")[1:]
        word_align_dict[uttid] = align_list
    return word_align_dict

def remove_sil_frame(encoder_out, encoder_out_lens, ratio_pad, phone_pad):
    ''' 
    logging.warning(f"encoder_out:{encoder_out.shape}{encoder_out}")
    logging.warning(f"encoder_out_lens:{encoder_out_lens.shape} {encoder_out_lens}")
    logging.warning(f"ratio_pad:{ratio_pad.shape} {ratio_pad}")
    logging.warning(f"phone_pad:{phone_pad}")
    ''' 
    len_phone = encoder_out.size(1)
    len_phone_frames = phone_pad.size(1)
    encoder_out_lens = encoder_out_lens.unsqueeze(1)
    num_elements_lists = torch.round(encoder_out_lens * ratio_pad) #[[2,3,4,0,0],[7,3,4,4,3],...]
    averaged_tensors = []
    utterance_averaged_tensors = []
    utterance_averaged_tensors_lengths = []
    num_phone_frames_tensors = []
    total_num_phone_frames_tensors = []
    for b in range(encoder_out.shape[0]):
        current_frame = 0
        t = 0
        for id, i in enumerate(num_elements_lists[b]):
            #if i == 0:
            #    averaged_tensor = encoder_out[b, current_frame, :].unsqueeze(0)
            #else:
            averaged_tensor = encoder_out[b, current_frame:int(i) + int(current_frame), :]
            if phone_pad[b][id] == 1 or phone_pad[b][id] == 0:
                current_frame = int(i)
                continue
            if t == 0:
                num_phone_frames_tensors = i.unsqueeze(0)
                averaged_tensors = averaged_tensor
                t = 1
            else:
                num_phone_frames_tensors = torch.cat((num_phone_frames_tensors, i.unsqueeze(0)), dim=0)
                averaged_tensors = torch.cat((averaged_tensors, averaged_tensor), dim=0)
            current_frame = int(i)
        #logging.warning(f"averaged_tensors:{averaged_tensors.shape}{averaged_tensors}")
        valid_phone_num = torch.tensor([averaged_tensors.size(0)])
        valid_phone_frames = torch.tensor([num_phone_frames_tensors.size(0)])
        if len_phone > valid_phone_num:
            padding = torch.zeros((len_phone-valid_phone_num,averaged_tensors.size(1)), dtype=averaged_tensors.dtype, device=averaged_tensors.device)
            averaged_tensors = torch.cat([averaged_tensors, padding], dim=0)
        if len_phone_frames > valid_phone_frames:
            padding = torch.full([len_phone_frames - valid_phone_frames], -1, dtype=num_phone_frames_tensors.dtype, device=num_phone_frames_tensors.device)
            num_phone_frames_tensors = torch.cat([num_phone_frames_tensors, padding], dim=0)
        if b == 0:
            utterance_averaged_tensors = averaged_tensors.unsqueeze(0)
            utterance_averaged_tensors_lengths = valid_phone_num
            total_num_phone_frames_tensors = num_phone_frames_tensors.unsqueeze(0)
        else:
            utterance_averaged_tensors = torch.cat((utterance_averaged_tensors, averaged_tensors.unsqueeze(0)), dim=0)
            utterance_averaged_tensors_lengths = torch.cat((utterance_averaged_tensors_lengths, valid_phone_num))
            total_num_phone_frames_tensors = torch.cat((total_num_phone_frames_tensors, num_phone_frames_tensors.unsqueeze(0)))
    
    return utterance_averaged_tensors, utterance_averaged_tensors_lengths, total_num_phone_frames_tensors
def combine_and_average(encoder_out, encoder_out_lens, num_elements_lists, len_phone):
    ''' 
    logging.warning(f"remove sil encoder_out:{encoder_out.shape}{encoder_out}")
    logging.warning(f"remove sil encoder_out_lens:{encoder_out_lens.shape} {encoder_out_lens}")
    logging.warning(f"remove sil num_elements_lists:{num_elements_lists.shape} {num_elements_lists}")
    logging.warning(f"len_phone:{len_phone}")
    ''' 
    
    encoder_out_lens = encoder_out_lens.unsqueeze(1)
    # num_elements_lists = ste_round(encoder_out_lens * ratio_pad) #[[2,3,4,0,0],[7,3,4,4,3],...]
    # logging.warning(f"num_elements_lists:{num_elements_lists}")
    averaged_tensors = []
    utterance_averaged_tensors = []
    utterance_averaged_tensors_lengths = []
    for b in range(encoder_out.shape[0]):
        current_frame = 0
        t = 0
        for id, i in enumerate(num_elements_lists[b]):
            if i == -1:
                break
            #if i == 0:
            #    averaged_tensor = encoder_out[b, current_frame, :].unsqueeze(0)
            #else:
            averaged_tensor = torch.mean(encoder_out[b, current_frame:int(i) + int(current_frame), :], dim=0, keepdim=True)
            #logging.warning(f"averaged_tensor:{averaged_tensor}")
            if t == 0:
                averaged_tensors = averaged_tensor
                t = 1
            else:
                averaged_tensors = torch.cat((averaged_tensors, averaged_tensor), dim=0)

            current_frame = int(i)
        #logging.warning(f"averaged_tensors:{averaged_tensors.shape}{averaged_tensors}")
        valid_phone_num = torch.tensor([averaged_tensors.size(0)])
        #logging.warning(f"valid_phone_num:{valid_phone_num}")
        if len_phone > valid_phone_num:
            padding = torch.zeros((len_phone-valid_phone_num,averaged_tensors.size(1)), dtype=averaged_tensors.dtype, device=averaged_tensors.device)
            averaged_tensors = torch.cat([averaged_tensors, padding], dim=0)
        if b == 0:
            utterance_averaged_tensors = averaged_tensors.unsqueeze(0)
            utterance_averaged_tensors_lengths = valid_phone_num
        else:
            #logging.warning(f"averaged_tensors:{averaged_tensors.shape} {averaged_tensors}")
            #logging.warning(f"utterance_averaged_tensors:{utterance_averaged_tensors.shape} {utterance_averaged_tensors}")
            utterance_averaged_tensors = torch.cat((utterance_averaged_tensors, averaged_tensors.unsqueeze(0)), dim=0)

            utterance_averaged_tensors_lengths = torch.cat((utterance_averaged_tensors_lengths, valid_phone_num))

    # utterance_averaged_tensors = torch.where(torch.isnan(utterance_averaged_tensors), torch.tensor(0.0).to(utterance_averaged_tensors.device), utterance_averaged_tensors)
    return utterance_averaged_tensors, utterance_averaged_tensors_lengths

def combine_and_average_word(phone_encoder_out, word_len_list, len_word):

    averaged_word_tensors = []
    batch_word_tensors = []
    batch_word_tensors_lengths = []
    for b in range(phone_encoder_out.shape[0]):
        current_num_phone = 0
        for i, num_phone_for_each_phone in enumerate(word_len_list[b]):

            averaged_word_tensor = torch.mean(phone_encoder_out[b, current_num_phone:current_num_phone + int(num_phone_for_each_phone), :], dim=0, keepdim=True)
            if i == 0:
                averaged_word_tensors = averaged_word_tensor
            else:
                averaged_word_tensors = torch.cat((averaged_word_tensors, averaged_word_tensor), dim=0)
            current_num_phone = int(num_phone_for_each_phone)
        valid_word_num = torch.tensor([averaged_word_tensors.size(0)])
        if len_word > valid_word_num:
            padding = torch.zeros((len_word - valid_word_num, averaged_word_tensors.size(1)), dtype=averaged_word_tensors.dtype, device=averaged_word_tensors.device)
            averaged_word_tensors = torch.cat([averaged_word_tensors, padding], dim=0)
        if b == 0:
            batch_word_tensors = averaged_word_tensors.unsqueeze(0)
            batch_word_tensors_lengths = valid_word_num
        else:
            batch_word_tensors = torch.cat((batch_word_tensors, averaged_word_tensors.unsqueeze(0)), dim=0)
            batch_word_tensors_lengths = torch.cat((batch_word_tensors_lengths, valid_word_num))
    return batch_word_tensors, batch_word_tensors_lengths
def remove_sil_and_merge_phone(encoder_out, encoder_out_lens, ratio_pad, phone_pad, len_phone):
    ''' 
    logging.warning(f"encoder_out:{encoder_out.shape}{encoder_out}")
    logging.warning(f"encoder_out_lens:{encoder_out_lens.shape} {encoder_out_lens}")
    logging.warning(f"ratio_pad:{ratio_pad.shape} {ratio_pad}")
    logging.warning(f"phone_pad:{phone_pad}")
    ''' 
    encoder_out_lens = encoder_out_lens.unsqueeze(1)
    num_elements_lists = torch.round(encoder_out_lens * ratio_pad) #[[2,3,4,0,0],[7,3,4,4,3],...]
    averaged_tensors = []
    phone_averaged_tensors = []
    phone_averaged_tensors_lengths = []
    for b in range(encoder_out.shape[0]):
        current_frame = 0
        t = 0
        for id, i in enumerate(num_elements_lists[b]):
            if i < 0: # padding
                assert phone_pad[b][id] == -1
                break
            if phone_pad[b][id] == 1 or phone_pad[b][id] == 0: # sil
                current_frame += int(i)
                continue
            if i == 0:
                averaged_tensor = encoder_out[b, current_frame, :].unsqueeze(0)
            else:
                averaged_tensor = torch.mean(encoder_out[b, current_frame:int(i) + int(current_frame), :], dim=0, keepdim=True) # (1,D)
            if t == 0:
                averaged_tensors = averaged_tensor
                t = 1
            else:
                averaged_tensors = torch.cat((averaged_tensors, averaged_tensor), dim=0)
            current_frame += int(i)


        #logging.warning(f"averaged_tensors:{averaged_tensors.shape}{averaged_tensors}")
        valid_phone_num = torch.tensor([averaged_tensors.size(0)])
        if len_phone > valid_phone_num:
            #padding = torch.full([len_phone-valid_phone_num,averaged_tensors.size(1)],-1,dtype=averaged_tensors.dtype, device=averaged_tensors.device)
            padding = torch.zeros((len_phone-valid_phone_num,averaged_tensors.size(1)), dtype=averaged_tensors.dtype, device=averaged_tensors.device)
            averaged_tensors = torch.cat([averaged_tensors, padding], dim=0)
        if b == 0:
            phone_averaged_tensors = averaged_tensors.unsqueeze(0)
            phone_averaged_tensors_lengths = valid_phone_num
        else:
            phone_averaged_tensors = torch.cat((phone_averaged_tensors, averaged_tensors.unsqueeze(0)), dim=0)
            phone_averaged_tensors_lengths = torch.cat((phone_averaged_tensors_lengths, valid_phone_num))
    
    return phone_averaged_tensors, phone_averaged_tensors_lengths

def combine_phone_to_word(phone_encoder_out, word_len_list, len_word):

    averaged_word_tensors = []
    batch_word_tensors = []
    batch_word_tensors_lengths = []
    for b in range(phone_encoder_out.shape[0]):
        current_num_phone = 0
        for i, num_phone_for_each_phone in enumerate(word_len_list[b]):

            averaged_word_tensor = torch.mean(phone_encoder_out[b, current_num_phone:current_num_phone + int(num_phone_for_each_phone), :], dim=0, keepdim=True)
            if i == 0:
                averaged_word_tensors = averaged_word_tensor
            else:
                averaged_word_tensors = torch.cat((averaged_word_tensors, averaged_word_tensor), dim=0)
            current_num_phone += int(num_phone_for_each_phone)
        valid_word_num = torch.tensor([averaged_word_tensors.size(0)])
        if len_word > valid_word_num:
            #padding = torch.full([len_word - valid_word_num, averaged_word_tensors.size(1)],-1, dtype=averaged_word_tensors.dtype, device=averaged_word_tensors.device)
            padding = torch.zeros((len_word - valid_word_num, averaged_word_tensors.size(1)), dtype=averaged_word_tensors.dtype, device=averaged_word_tensors.device)
            averaged_word_tensors = torch.cat([averaged_word_tensors, padding], dim=0)
        if b == 0:
            batch_word_tensors = averaged_word_tensors.unsqueeze(0)
            batch_word_tensors_lengths = valid_word_num
        else:
            batch_word_tensors = torch.cat((batch_word_tensors, averaged_word_tensors.unsqueeze(0)), dim=0)
            batch_word_tensors_lengths = torch.cat((batch_word_tensors_lengths, valid_word_num))
    return batch_word_tensors, batch_word_tensors_lengths
 

def calculate_pcc(encoder_out, score):  # (B, num), (B, num)
    encoder_out, score = encoder_out.cpu().detach(), score.cpu().detach()
    valid_token_pred = []
    valid_token_target = []
    for i in range(encoder_out.shape[0]):
        for j in range(encoder_out.shape[1]):
            if score[i,j] >= 0:
                valid_token_pred.append(encoder_out[i, j])
                valid_token_target.append(score[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)
    #logging.warning(f"valid_token_target:{valid_token_target}")
    #logging.warning(f"valid_token_pred:{valid_token_pred}")
    return np.corrcoef(valid_token_pred, valid_token_target)[0,1], valid_token_pred, valid_token_target


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ESPnetGOPModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = False,
        report_wer: bool = False,
        output_dim: int = 512,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        utt2align: str = "utt2align",
        word_len_align: str = "word_len_align",
        adim: int = 512,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        '''
        Gop features related 
        '''
        self.utt2align = utt2align
        self.word_len_align = word_len_align
        self.adim = adim

        self.output_dim = output_dim
        self.phone_adapter = Adapter(self.output_dim, self.output_dim // 2, 1)
        self.word_adapter = Adapter(self.output_dim, self.output_dim// 2, 1)
        self.utt_adapter = Adapter(self.output_dim, self.output_dim // 2, 1) 
        
        self.phone_dict = get_phone_dict(self.utt2align)
        self.word_len_align_dict = get_word_len_align(self.word_len_align)
        self.mseloss = nn.MSELoss()


        self.error_calculator = None

        self.ctc = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def mean_std_pooling(self, output, word_score):
        mask = (word_score != -1).float().unsqueeze(-1).expand(word_score.size(0), word_score.size(1),output.size(-1))
        mean = torch.sum(output * mask, dim=1) / torch.sum(mask, dim=1)
        sq_diff = torch.sum((output - mean.unsqueeze(1))**2 * mask, dim=1)
        count_valid = torch.sum(mask, dim=1) - 1
        std = torch.sqrt(sq_diff / count_valid)
        return mean, std

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        phone_score: torch.Tensor,
        phone_score_lengths: torch.Tensor,
        word_score: torch.Tensor,
        word_score_lengths: torch.Tensor,
        utt_score: torch.Tensor,
        utt_score_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            phone_score: (Batch, T2)
            phone_score_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == phone_score.shape[0]
            == phone_score_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape, phone_score.shape, phone_score_lengths.shape)
        batch_size = speech.shape[0]
        # for data-parallel
        text = text[:, : text_lengths.max()]
        uttid = kwargs["utt_id"]
        #logging.warning(f"uttid: {uttid}")
        mode = kwargs["mode"]
        
        phone_ratio = []
        phone_list = []
        word_len_list = []
        for uttid in kwargs["utt_id"]:
            phone_ratio.append(self.phone_dict[uttid]["phone_ratio"])
            phone_list.append(self.phone_dict[uttid]["phone_list"])
            word_len_list.append(self.word_len_align_dict[uttid])

        
        # phone_out, word_out, utt_out, encoder_out_lengths = self.encode(speech, speech_lengths, phone_ratio, phone_list) # (B,T,D)

        phone_out, word_out, utt_out, encoder_out_lengths = self.encode(speech, speech_lengths) # (B,T,D)
        ratio_pad = pad_list(phone_ratio, -1, encoder_out=phone_out)
        phone_pad = pad_list(phone_list, -1, encoder_out=phone_out)
        
        e_phone_out, e_phone_out_lengths = remove_sil_and_merge_phone(phone_out, encoder_out_lengths, ratio_pad, phone_pad, phone_score.size(1))

        #logging.warning(f"e_phone_out: {e_phone_out.shape} {e_phone_out}")
        #logging.warning(f"e_phone_out_lengths: {e_phone_out_lengths}")
        e_phone_adapter_out = F.relu(self.phone_adapter(e_phone_out)) # (B, num_phone, 1)

        # word part
        word_e_phone_out, word_e_phone_out_lengths = remove_sil_and_merge_phone(word_out, encoder_out_lengths, ratio_pad, phone_pad, phone_score.size(1))
        e_word_out, e_word_out_lengths = combine_phone_to_word(word_e_phone_out, word_len_list, word_score.size(1))
        e_word_adapter_out = F.relu(self.word_adapter(e_word_out))

        # utt part
        utt_out_mean = torch.mean(utt_out, dim=1)
        e_utt_adapter_out = F.relu(self.utt_adapter(utt_out_mean))
        
        mask_phone = (phone_score>=0)
        phone_hyp = e_phone_adapter_out.squeeze(2) * mask_phone
        phone_score_0 = phone_score * mask_phone
        phn_mseloss = self.mseloss(phone_hyp, phone_score_0)
        phn_mseloss = phn_mseloss * (mask_phone.shape[0] * mask_phone.shape[1]) / torch.sum(mask_phone)
        phn_pcc, valid_phone_pred, valid_phone_true = calculate_pcc(phone_hyp, phone_score)

        mask_word = (word_score>=0)
        word_hyp = e_word_adapter_out.squeeze(2) * mask_word
        word_score_0 = word_score * mask_word
        word_mseloss = self.mseloss(word_hyp, word_score_0)
        word_mseloss = word_mseloss * (mask_word.shape[0] * mask_word.shape[1] ) / torch.sum(mask_word)
        word_pcc, valid_word_pred, valid_word_true = calculate_pcc(word_hyp, word_score)

        utt_mseloss = self.mseloss(e_utt_adapter_out, utt_score)
        utt_pred = e_utt_adapter_out.view(-1).cpu().detach().numpy()
        utt_true = utt_score.view(-1).cpu().detach().numpy()
        utt_pcc = np.corrcoef(utt_pred, utt_true)[0,1]

        stats = dict()
        stats["phn_mseloss"] = phn_mseloss.detach()
        stats["word_mseloss"] = word_mseloss.detach()
        stats["utt_mseloss"] = utt_mseloss.detach()

        stats["phn_pcc"] = phn_pcc
        stats["word_pcc"] = word_pcc
        stats["utt_pcc"] = utt_pcc

        Loss = phn_mseloss + word_mseloss + utt_mseloss
        #Loss = word_mseloss + utt_mseloss
        #Loss = utt_mseloss
        stats["total_loss"] = Loss

        loss, stats, weight = force_gatherable((Loss, stats, batch_size), Loss.device)
        '''         
        logging.warning(f"phone_hyp:{phone_hyp.shape} {phone_hyp}")
        logging.warning(f"phone_score_0:{phone_score_0.shape} {phone_score_0}")
        logging.warning(f"word_hyp:{word_hyp.shape} {word_hyp}")
        logging.warning(f"word_score_0:{word_score_0.shape} {word_score_0}")
        logging.warning(f"utt_pred: {utt_pred.shape} {utt_pred}")
        logging.warning(f"utt_true: {utt_true.shape} {utt_true}")
        '''
        
        
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        retval = dict()
        retval["stats"] = stats
        retval["weight"] = weight
        if mode == "valid":
            retval["phone_pred"] = valid_phone_pred
            retval["phone_true"] = valid_phone_true
            retval["word_pred"] = valid_word_pred
            retval["word_true"] = valid_word_true
            retval["utt_pred"] = utt_pred
            retval["utt_true"] = utt_true
            return retval
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        '''
        with autocast(False):
            #logging.warning(f"feats:{feats.shape} {feats}") 
            #logging.warning(f"feats_lengths:{feats_lengths}")
            """ According to the alignment, remove the silence frame """
            #ratio_pad = pad_list(phone_ratio, 0, feats_out=feats)
            #phone_pad = pad_list(phone_list, 0, feats_out=feats)
            #feats, feats_lengths, num_phone_frames = remove_sil_frame(feats, feats_lengths, ratio_pad, phone_pad)
            #logging.warning(f"feats shape:{feats.shape}")
            #logging.warning(f"feats_lengths : {feats_lengths.shape}{feats_lengths}")


            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
        '''
        
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        phone_out, word_out, utt_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        #logging.warning(f"encoder_out shape:{encoder_out.shape}")
        #logging.warning(f"after encoder_out_lens shape:{encoder_out_lens.shape} {encoder_out_lens}")


        assert phone_out.size(0) == speech.size(0), (
            phone_out.size(),
            speech.size(0),
        )
        assert phone_out.size(1) <= encoder_out_lens.max(), (
            phone_out.size(),
            encoder_out_lens.max(),
        )


        return phone_out, word_out, utt_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    
