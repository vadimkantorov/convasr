import time
from multiprocessing import Queue, Pool
import multiprocessing as mp
import subprocess

bash_template = "python3 run.py --arg {arg}"

bash_template = """
/miniconda/bin/python3  train.py \
    --githttp https://github.com/vadimkantorov/convasr/commit/%h \
    --verbose --lang ru \
    --model JasperNetBig \
    --train-batch-size 16 --val-batch-size 256 \
    --scheduler MultiStepLR --decay-milestones 40000 \
    --lr 1e-2 \
    --optimizer NovoGrad \
    --train-data-path {train_set} \
    --val-data-path data/domain_splits/mixed_val.csv data/domain_splits/clean_val.csv {val_set} \
    --analyze kontur_calls_micro.csv \
    --val-iteration-interval 2500 \
    --fp16 O2 \
    --checkpoint {cp} \
    --experiment-name finetune_{i} \
    --epochs {epoch} 
"""

def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    (i, train_set, val_set, cp) = args
    print(train_set)
    print(val_set)
    print(cp)
    epoch = int(cp.split('_')[-2].replace('epoch', ''))
    print(epoch) 
    bashcmd = bash_template.format(i=i, epoch=epoch+20, train_set='data/kfold_splits/' + train_set, val_set = 'data/kfold_splits/' + val_set, cp='data/experiments/' + cp)
    print(bashcmd)
    with open(f'data/experiments/experiment_{i}_output.txt', 'a+') as out:
        with open(f'data/experiments/subprocess_{i}_exit_codes.txt', 'a+') as experiment_results:
            exit_code = subprocess.call(bashcmd.split(),
                                        stdout=out,
                                        stderr=out,
                                        env={"CUDA_VISIBLE_DEVICES": f"{get_process_to_gpu_mapping()[worker_id]}"}, )

            experiment_results.write(f"{i}\t{exit_code}\n")


def worker_loop(queue):
    while not queue.empty():
        task = queue.get()
        process = mp.current_process()
        worker_id = process.name.split('-')[1]

        print(f'worker id: {worker_id}, run new task: {task}')
        run_task(worker_id, task)

        time.sleep(30)
        if queue.empty():
            break


def main():
    queue = Queue()

    for v in [
    (0, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_books_normalized_train/checkpoint_epoch30_iter0060000.pt'),
    (1, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_books_radio_normalized_train/checkpoint_epoch36_iter0060000.pt'),
    (2, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_books_youtube_normalized_train/checkpoint_epoch21_iter0060000.pt'),
    (3, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_books_youtube_radio_normalized_train/checkpoint_epoch26_iter0060000.pt'),
    (4, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_clean_train/checkpoint_epoch45_iter0003645.pt'),
    (5, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_radio_normalized_train/checkpoint_epoch44_iter0060000.pt'),
    (6, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_youtube_normalized_train/checkpoint_epoch16_iter0060000.pt'),
    (7, 'trainset_kfold_05022020_fold_1.csv', 'valset_kfold_05022020_fold_1.csv', 'JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comp_youtube_radio_normalized_train/checkpoint_epoch24_iter0060000.pt')
	]:
        queue.put(v)

    pool = Pool(4, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
