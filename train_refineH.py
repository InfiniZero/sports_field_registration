import os
import opts
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lib.model_refineH import SFRNet
from lib.dataset_refineH import ImageDataset
from lib.loss_function import refine_h_loss_func


def main(opt):
    torch.cuda.set_device(0)
    train(opt)


def train(opt):
    print("load model")
    model = SFRNet(opt)
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    checkpoint_resume = opt["weights_path"] + "/unet.pth"
    if os.path.isfile(checkpoint_resume):
        checkpoint = torch.load(checkpoint_resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    start_epoch = 0
    opt["mode"] = "train"
    train_data = ImageDataset(opt)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=opt["batch_size"], shuffle=True)
    opt["mode"] = "test"
    test_data = ImageDataset(opt)
    test_loader = DataLoader(dataset=test_data,
                            batch_size=opt["batch_size"], shuffle=False)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    best_loss = 5
    for epoch in range(start_epoch, opt["train_epochs"]):
        print("Training...")
        print("lr: ", optimizer.param_groups[0]['lr'])
        train_net(train_loader, model, optimizer, epoch)
        print("Testing...")
        best_loss = test_net(test_loader, model, epoch, best_loss)
        scheduler.step()


def train_net(data_loader, model, optimizer, epoch):
    model.train()
    step_loss = 0
    epoch_loss = 0
    deltah_loss = 0
    iou_loss = 0
    for step, sample_batch in enumerate(data_loader):
        input_data = sample_batch["image"].cuda()
        temp_data = sample_batch["temp"].cuda()
        deltaH_data = sample_batch["deltaH"].cuda()
        iou_data = sample_batch["iou"].cuda()
        output_deltaH, output_iou = model(input_data, temp_data)
        loss = refine_h_loss_func(
            output_deltaH, deltaH_data, output_iou, iou_data)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        step_loss += loss[0].item()
        epoch_loss += loss[0].item()
        deltah_loss += loss[1].item()
        iou_loss += loss[2].item()
        del input_data, temp_data, loss
        torch.cuda.empty_cache()
        if step % 10 == 0:
            if step == 0:
                step_loss = 0
                continue
            step_loss = step_loss/10
            deltah_loss = deltah_loss/10
            iou_loss = iou_loss/10
            print('Epoch: ', epoch + 1, 'Step: ', step,
                  '| train_loss: %.4f' % step_loss,
                  ' deltah_loss: %.4f' % deltah_loss,
                  ' iou_loss: %.4f' % iou_loss)
            step_loss = 0
            deltah_loss = 0
            iou_loss = 0
    epoch_loss = epoch_loss/(step+1)
    print('Epoch: ', epoch + 1, '| train_loss: %.4f' % epoch_loss)


def test_net(data_loader, model, epoch, best_loss):
    model.eval()
    epoch_loss = 0
    step_loss = 0
    deltah_loss = 0
    iou_loss = 0
    for step, sample_batch in enumerate(data_loader):
        input_data = sample_batch["image"].cuda()
        temp_data = sample_batch["temp"].cuda()
        deltaH_data = sample_batch["deltaH"].cuda()
        iou_data = sample_batch["iou"].cuda()
        with torch.no_grad():
            output_deltaH, output_iou = model(input_data, temp_data)
        loss = refine_h_loss_func(
            output_deltaH, deltaH_data, output_iou, iou_data)
        step_loss += loss[0].item()
        epoch_loss += loss[0].item()
        deltah_loss += loss[1].item()
        iou_loss += loss[2].item()
        del input_data, temp_data, loss
        torch.cuda.empty_cache()
        if step % 10 == 0:
            if step == 0:
                step_loss = 0
                continue
            step_loss = step_loss/10
            deltah_loss = deltah_loss/10
            iou_loss = iou_loss/10
            print('Epoch: ', epoch + 1, 'Step: ', step,
                  '| test_loss: %.4f' % step_loss,
                  ' deltah_loss: %.4f' % deltah_loss,
                  ' iou_loss: %.4f' % iou_loss)
            step_loss = 0
            deltah_loss = 0
            iou_loss = 0
    epoch_loss = epoch_loss/(step+1)
    print('Epoch: ', epoch + 1, '| test_loss: %.4f' % epoch_loss)
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        print("best_loss: ", best_loss)
        torch.save(state, opt["weights_path"] + "/refineH.pth")
    return best_loss


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt['training_lr'] = 1e-5
    opt['batch_size'] = 4
    opt['step_size'] = 500
    opt['train_epochs'] = 500
    main(opt)
