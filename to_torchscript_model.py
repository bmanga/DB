import os
import sys
import torch
import argparse
import numpy as np

# import net_model
from concern.config import Configurable, Config


# def load_model(bpth_file_path):
#     cuda = torch.cuda.is_available()
#     device = torch.device('cuda') if cuda else torch.device('cpu')
#     model = net_model.load_model(bpth_file_path, device)
#     model.eval()
#     return model


# def main(bpth_file_path):
def main():
    parser = argparse.ArgumentParser(description='Text Detection: TorchScript Model Creator')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    # Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    experiment.load('evaluation', **experiment_args)
    # args = cmd
    # model_saver = experiment.train.model_saver
    structure = experiment.structure
    model_path = args['resume']

    # Use gpu or not
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
    
    model = structure.builder.build(device)
    
    if not os.path.exists(model_path):
        print("Checkpoint not found: " + model_path)
        return
    print("Resuming from " + model_path)
    states = torch.load(model_path, map_location=device)
    model.load_state_dict(states, strict=False)
    print("Resumed from " + model_path)

    model.eval()

    script_module = torch.jit.script(model)

    file_path_without_ext = os.path.splitext(bpth_file_path)[0]
    output_file_path = file_path_without_ext + ".pt"
    script_module.save(output_file_path)
    print("TorchScript model created:", output_file_path)

    # model = load_model(bpth_file_path)
    # script_module = torch.jit.script(model)

    # file_path_without_ext = os.path.splitext(bpth_file_path)[0]
    # output_file_path = file_path_without_ext + ".pt"
    # script_module.save(output_file_path)
    # print("TorchScript model created:", output_file_path)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Invalid arguments!")
    #     print("Usage: python3 " + sys.argv[0] + " <bpth_file_path>")
    # else:
    #     main(sys.argv[1])
    main()
