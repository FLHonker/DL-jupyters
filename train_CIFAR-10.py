# CIFAR-10 通用训练python脚本
# --------------------------------------
import torch 
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse

# start
if __name__ == '__main__':
    main()

# main
def main():
    parser = argparse.ArgumentParser(description='cifar-10 with PyTorch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch tp train for') 
    parser.add_argument('--trainBatchSize', default=128, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=128, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use cuda or not')
    
    config_list = ['--lr', '0.001', '--epoch', '50', '--trainBatchSize', '128', '--testBatchSize', '128', '--cuda', 'True']
    args = parser.parse_args(config_list) 
    
    solver = Solver(args)
    solver.run()

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Solver
class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = 'cuda' if config.cuda else 'cpu'
        self.train_loader = None
        self.test_loader = None
        
    def print_model(self):
        print(self.model)
        
    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        self.train_loader = DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.test_batch_size, shuffle=False)
    
    def load_model(self):
        # self.model = LeNet().to(self.device)
        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        self.model = GoogLeNet().to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        # self.model = WideResNet(depth=28, num_classes=10).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    # train
    def train(self):
        print('Training:')
        self.model.train()
        train_loss = 0.0
        train_correct = 0 
        total = 0 
        
        for ibatch, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1) # second param "1" represents the dimension to be reduced
            total += labels.size(0)
            
            # train_correct incremented by one if predicted right
            # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            train_correct += (pred == labels).sum().item()
            if ibatch % 99 == 0:
                print('\t{}/{}: loss = {:.4f}, Acc = {:.3f}%'.format(ibatch, len(self.train_loader), train_loss/(ibatch+1), 100. * train_correct/total))
        return train_loss, float(train_correct/total)
    
    # test
    def test(self):
        print('Testing:')
        self.model.eval()
        test_loss = 0.0 
        test_correct = 0 
        total = 0
        
        with torch.no_grad():
            for ibatch, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                test_correct += (pred == labels).sum().item()
                if ibatch % 99 == 0:
                    print('\t{}/{}: loss = {:.4f}, Acc = {:.3f}%'.format(ibatch, len(self.test_loader), test_loss/(ibatch+1), 100. * test_correct/total))
        return test_loss, float(test_correct/total)
    
    def save_model(self):
        model_out_path = './model/vgg_cifar10.pth'
        torch.save(self.model, model_out_path)
        print("* Checkpoint saved to {}".format(model_out_path))
        
    # run
    def run(self):
        self.load_data()
        self.load_model() 
        accuracy = 0.
        
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: {}/{}".format(epoch, self.epochs))
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
        print("===> BEST ACC. PERFORMANCE: {:.3f}%".format(accuracy * 100))
        self.save_model()