#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

NETID=dec415
source activate /scratch/${NETID}/nlu/env

# remove old files
rm -rf /scratch/${NETID}/nlu/data
rm -rf /scratch/${NETID}/nlu/code

# specify where the data is stored
GLUE_DIR=/scratch/${NETID}/nlu/data

# store the data
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip
unzip BoolQ.zip -d ${GLUE_DIR}

cd /scratch/${NETID}/nlu
mkdir code
cd code
git clone https://github.com/danielchain3/transformers.git
cd transformers

TASK_NAME=BoolQ
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH
python ./examples/run_glue.py \
           --model_type roberta \
           --model_name_or_path roberta-base \
           --task_name ${TASK_NAME} \
           --do_train \
           --do_eval \
           --do_lower_case \
           --data_dir ${GLUE_DIR}/${TASK_NAME} \
           --max_seq_length 128 \
           --per_gpu_eval_batch_size=8 \
           --per_gpu_train_batch_size=8 \
           --learning_rate 2e-5\
           --num_train_epochs 3.0 \
           --output_dir /scratch/${NETID}/nlu/${TASK_NAME}_run/

