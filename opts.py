import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--weights_path',
        type=str,
        default='./weights')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=60)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--step_size',
        type=int,
        default=40)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    # Overall Dataset settings
    parser.add_argument(
        '--img_train_dir',
        type=str,
        default="./dataset/train/img")
    parser.add_argument(
        '--H_train_dir',
        type=str,
        default="./dataset/train/H")
    parser.add_argument(
        '--mask_train_dir',
        type=str,
        default="./dataset/train/mask")
    parser.add_argument(
        '--img_test_dir',
        type=str,
        default="./dataset/test/img")
    parser.add_argument(
        '--H_test_dir',
        type=str,
        default="./dataset/test/H")
    parser.add_argument(
        '--H_mat_test_dir',
        type=str,
        default="./dataset/test/H_mat")
    parser.add_argument(
        '--mask_test_dir',
        type=str,
        default="./dataset/test/mask")
    parser.add_argument(
        '--field_template_file',
        type=str,
        default="./dataset/field_nba_new.jpg")
    parser.add_argument(
        '--test_image_file',
        type=str,
        default="./demo/demo.jpg")
    parser.add_argument(
        '--field_view_file',
        type=str,
        default="./demo/field_view.jpg")
    parser.add_argument(
        '--broadcast_view_file',
        type=str,
        default="./demo/broadcast_view.jpg")

    args = parser.parse_args()

    return args
