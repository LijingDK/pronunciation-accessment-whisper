
#export train_cmd="slurm.pl --quiet --exclude=node0[6]"
#export decode_cmd="slurm.pl --quiet"
#export mkgraph_cmd="slurm.pl --quiet --exclude=node0[3-9]"
#export cuda_cmd="slurm.pl --quiet --nodelist=node03 --exclude=node0[3,8]"
#--num-threads 4 
#export cuda_cmd="slurm.pl --quiet --exclude=node0[1,2,3]" 
#export cuda_cmd="slurm.pl --quiet" 
export train_cmd="utils/run.pl"
export decode_cmd="utils/run.pl"
export mkgraph_cmd="utils/run.pl"
export cuda_cmd="utils/run.pl"




