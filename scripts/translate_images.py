#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image

from uvcgan2.consts import (MERGE_NONE, ROOT_DATA)
from uvcgan2.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from uvcgan2.torch.select import extract_name_kwargs

from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as images'
    )
    add_standard_eval_parsers(parser)
    add_plot_extension_parser(parser)

    return parser.parse_args()

def save_images(model, savedir, sample_counter, ext, domain, datasets_config):
    #items은 real_a, fake_b이런거 model은 UVCGAN2 class의 모델
    #savedir은 저장할 공간 : real_a, fake_b 전까지, ext는 확장자
    name, kwargs = extract_name_kwargs(datasets_config[domain].dataset)
    domain_name = kwargs.pop('domain')
    orgdir         = os.path.join(ROOT_DATA, kwargs.pop('path', name))
    domain_path = orgdir + "/test/" + domain_name
    image_names = os.listdir(domain_path)
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]

            image = tensor_to_image(torch_image[index])
            image = np.round(255 * image).astype(np.uint8)
            image = Image.fromarray(image)
            image_name = sorted(image_names)[sample_index][:-4]
            path  = os.path.join(savedir, name, f'{image_name}(sample_{sample_index})')
            for e in ext:
                image.save(path + '.' + e)

            sample_counter[name] += 1

def dump_single_domain_images(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext, datasets_config
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Translating domain {domain}'

    for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
        model.set_input(batch, domain = domain)
        model.forward_nograd()

        save_images(model, savedir, sample_counter, ext, domain, datasets_config)

def dump_images(model, data_list, n_eval, batch_size, savedir, ext, datasets_config):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir) #그냥 real, fake 등  폴더 만드는 코드

    sample_counter = collections.defaultdict(int)
    if isinstance(ext, str):
        ext = [ ext, ]

    for domain, data_it in enumerate(data_list):
        # print(data_list, domain, data_it)
        # data_list는 dataloader2개, domain은 0또는 1로 나타나고 dataloader 중 0번째와 1번째, data_it은 dataloader 1개 
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, ext, datasets_config
        )

def main():
    cmdargs = parse_cmdargs()

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE
    )
    #data_list로 Dataloader object가 2개 들어가 있음, evaldir도 결국 저장될 위치임
    if not isinstance(data_list, (list, tuple)):
        data_list = [ data_list, ]

    savedir = get_eval_savedir(
        evaldir, 'images', cmdargs.model_state, cmdargs.split
    )
    datasets_config = args.config.data.datasets

    dump_images(
        model, data_list, cmdargs.n_eval, args.batch_size, savedir,
        cmdargs.ext, datasets_config
    )

if __name__ == '__main__':
    main()

