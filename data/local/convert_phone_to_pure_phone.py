

phone_to_pure_phone = "/home3/lirui001/w2022/github/espnet/egs2/timit/lm/data/local/phone-to-pure-phone.int"
dict_phone_to_pure_phone = {}
with open(phone_to_pure_phone, "r") as f:
    lines = f.readlines()

for line in lines:
    phone, pure_phone = line.strip().split(" ")

    dict_phone_to_pure_phone[phone] = pure_phone


with open("ali_train.txt","r") as f:
    lines = f.readlines()
with open("pure_phone_ali_train.txt", "w") as F:
    for line in lines:
        uttid = line.strip().split(" ")[0]
        frames = line.strip().split(" ")[1:]
        F.write(uttid)
        for frame in frames:
            F.write(" " + str(dict_phone_to_pure_phone[frame]))
        F.write("\n")
        
