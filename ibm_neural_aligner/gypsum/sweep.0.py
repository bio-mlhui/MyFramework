# This script creates and optionally runs an experiment sweep for training and eval the aligner.

import argparse
import collections
import copy
import json
import os

from templates import template, eval_template, eval_dist_template

parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
parser.add_argument('--launch-eval', action='store_true')
args = parser.parse_args()

prefix = '2021-11-05a'

def default_model_cfg():
    cfg = {}
    cfg['text_emb'] = 'char'
    cfg['text_enc'] = 'bilstm'
    cfg['text_project'] = 200
    cfg['amr_emb'] = 'char'
    cfg['amr_enc'] = 'lstm' #
    cfg['amr_project'] = 200
    cfg['dropout'] = 0.3 #
    cfg['context'] = 'xy'
    cfg['hidden_size'] = 200
    cfg['prior'] = 'attn'
    cfg['output_mode'] = 'tied'
    cfg['num_amr_layers'] = 2
    return cfg

def default_flags():
    flags = {}
    flags['lr'] = 2e-3
    flags['mask'] = 0 #
    return flags

def render_flags(flags):
    flags_str = ''

    for k, v in flags.items():
        if isinstance(k, bool):
            if v:
                flags_str += ' --{}'.format(v)

        else:
            flags_str += ' --{} {}'.format(k, v)

    return flags_str

def main():
    exp_list =[]

    num_seeds = 2

    c_exp = collections.Counter()

    for i in range(num_seeds):
        for task in ['amr2']:
            for model in ['lstm', 'bilstm', 'gcn', 'gcn_gated']:
                for lr_ in ['lrA', 'lrB']:
                    for mask_ in ['maskA', 'maskB']:
                        for drop_ in ['dropA', 'dropB']:
                            sofar = len(exp_list)

                            if sofar % 3 <= 1:
                                partition = '1080ti-long'
                            else:
                                partition = '2080ti-long'

                            lr = {'lrA': 2e-3, 'lrB': 1e-4}[lr_]
                            mask = {'maskA': 0, 'maskB': 0.15}[mask_]
                            dropout = {'dropA': 0.1, 'dropB': 0.3}[drop_]

                            flags = default_flags()
                            flags['lr'] = lr
                            flags['mask'] = mask
                            flags_str = render_flags(flags)

                            model_cfg = default_model_cfg()
                            model_cfg['amr_enc'] = model
                            model_cfg['dropout'] = dropout
                            model_cfg_str = " --model-config '{}'".format(json.dumps(model_cfg))

                            exp_key = '{}.{}.{}.{}.{}'.format(task, model, lr_, drop_, mask_)
                            exp_id = c_exp[exp_key]
                            c_exp[exp_key] += 1

                            d = {}
                            d['exp_id'] = exp_id
                            d['exp_key'] = exp_key
                            d['prefix'] = prefix
                            d['partition'] = partition
                            d['task'] = task
                            d['flags'] = flags_str
                            d['model_cfg'] = model_cfg_str

                            # TRAIN
                            name = 'gypsum.{prefix}.{exp_key}.{exp_id}'.format(**d)
                            d['name'] = name

                            os.system('mkdir -p log/{}'.format(name))

                            script = template.format(**d)

                            path = 'log/{}/script.sh'.format(name)

                            with open(path, 'w') as f:
                                f.write(script)

                            ex = {}
                            ex['name'] = name
                            ex['script'] = script
                            ex['script_path'] = path
                            ex['log_path'] = 'log/{}'.format(name)
                            exp_list.append(ex)

                            print(script)

                            # EVAL

                            if sofar % 3 <= 1:
                                partition = '1080ti-short'
                            else:
                                partition = '2080ti-short'

                            ex['eval_info'] = []

                            base_d = d
                            base_name = name

                            for i_exp in range(2):
                                for model_epoch in ['epoch_100', 'epoch_200', 'epoch_300', 'latest']:

                                    d = copy.deepcopy(base_d)
                                    d['i_exp'] = i_exp

                                    if i_exp == 1:
                                        if not 'gcn' in ex['name']:
                                            continue
                                        d['flags'] = d['flags'] + ' --mask-at-inference'

                                    d['partition'] = partition
                                    d['model_epoch'] = model_epoch
                                    d['load'] = './log/{}/model.{}.pt'.format(base_name, model_epoch)
                                    name = 'gypsum.{prefix}.{exp_key}.{exp_id}.{i_exp}.eval.{model_epoch}'.format(**d)
                                    d['name'] = name

                                    os.system('mkdir -p log/{}'.format(name))
                                    script = eval_template.format(**d)
                                    path = 'log/{}/script.sh'.format(name)
                                    with open(path, 'w') as f:
                                        f.write(script)

                                    info = {}
                                    info['script'] = script
                                    info['script_path'] = path
                                    info['log_path'] = './log/{}'.format(name)
                                    ex['eval_info'].append(info)

                                    print('eval', path)

                                    # write align dist
                                    d = copy.deepcopy(d)
                                    name = 'gypsum.{prefix}.{exp_key}.{exp_id}.{i_exp}.eval_align_dist.{model_epoch}'.format(**d)
                                    d['name'] = name

                                    os.system('mkdir -p log/{}'.format(name))
                                    script = eval_dist_template.format(**d)
                                    path = 'log/{}/script.sh'.format(name)
                                    with open(path, 'w') as f:
                                        f.write(script)

                                    info['align_dist_script_path'] = path


    if args.launch:
        for exp in exp_list:
            os.system('sbatch {}'.format(exp['script_path']))


    if args.launch_eval:
        for exp in exp_list:
            for info in exp['eval_info']:
                eval_json = os.path.join(info['log_path'], 'train.txt.no_wiki.eval.json')
                if os.path.exists(eval_json):
                    print('skip', eval_json)
                    continue
                path = info['script_path']
                os.system('sbatch {}'.format(path))

    print(len(exp_list))

    file_list_path = 'eval_json.{}.txt'.format(prefix)
    with open(file_list_path, 'w') as f:
        for exp in exp_list:
            for info in exp['eval_info']:
                f.write('{} {}\n'.format(info['log_path'], exp['log_path']))
    print(file_list_path, sum([len(exp['eval_info']) for exp in exp_list]))


main()


