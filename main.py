#from models import *
#from try_models_advanced import *
from try_models import *
import LoadData
from torch.utils.data import Dataset, DataLoader
import sys
import argparse

if __name__ == "__main__":
    # ##"Learning Alignment for Multimodal Emotion Recognition from Speech"的不完全实现
    # batch_data_train = LoadData.LoadSenData("E:\\NLP\SpeechEmotion\dump\\all\\train", balance=True)
    # train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=True, shuffle=True)
    # batch_data_test = LoadData.LoadSenData("E:\\NLP\SpeechEmotion\dump\\all\\eval")
    # test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=True, shuffle=True)
    # model = MultiModel().to(device)
    # lr, num_epochs = 0.01, epochs
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    #
    # train_and_test(train_loader,test_loader,model, criterion, optimizer, num_epochs)
    #
    # net_NN = MultiModel().cuda()
    # torch.save(net_NN, '.\\name.pkl')
    # net = torch.load('.\\name.pkl')

    ##"Context-Dependent Domain Adversarial Neural Network for Multimoda Emotion Recognition"
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default="tmp.txt", help='output confusion matrix to a file')
    parser.add_argument('--classify', type=str, default="emotion", help='choose train vad or emotion')
    parser.add_argument('--modal', type=str, default="audio", help='choose "text","audio","multi"')
    parser.add_argument('--fusion', type=str, default="AT",
                        help='choose "AT_fusion" "Concat" "ADD",or "ADD" "Dot" in try models')
    parser.add_argument('--dataset', type=str, default="ground_truth",
                        help='choose "google_cloud" "speech_recognition" "ground_truth" "resources" or "v" "a" "d"')
    parser.add_argument('--criterion', type=str, default="CrossEntropyLoss", help='choose "MSELoss" "CrossEntropyLoss"')
    parser.add_argument('--loss_delta', type=float, default=0, help='change loss proportion')
    args = parser.parse_args()
    classify = args.classify  # "emotion" "vad"
    modal = args.modal  # "text","audio","multi"
    fusion = args.fusion  # "AT_fusion" "Concat" "ADD"
    dataset = args.dataset  # "google cloud" "speech recognition" "ground truth" "v" "a" "d"
    criterion = args.criterion  # "MSELoss" "CrossEntropyLoss"
    # matrix_save_file=sys.argv[1]
    batch_size=20
    batch_data_train = LoadData.LoadDiaData('train', dataset, classify)
    # batch_data_train = LoadData.LoadDiaData_4('train')
    train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=False, shuffle=False)
    # batch_data_test = LoadData.LoadDiaData_4('test')
    batch_data_test = LoadData.LoadDiaData('test', dataset, classify)
    test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False)

    # model = Multilevel_Multiple_Attentions(modal, fusion).to(device)
    # model = After_fusion(fusion).to(device)
    # model = InAdvance_fusion(fusion).to(device)
    # model =Middle_fusion(fusion).to(device)
    model = DIDIlike_fusion(fusion).to(device)
    lr, num_epochs = 1e-3, epochs

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

    # criterion = nn.CrossEntropyLoss(reduction='none')

    #train_and_test_Multilevel(train_loader, test_loader, model, criterion, optimizer, num_epochs, args.outfile)
    #try_models.train_and_test_afterfusion(train_loader, test_loader, model, criterion, optimizer, num_epochs,args.outfile)
    # train_and_test_afterfusion(train_loader, test_loader, model, criterion, optimizer, num_epochs, args.loss_delta,args.outfile)
    #train_and_test_afterfusion(train_loader, test_loader, model, criterion, optimizer, num_epochs,args.outfile)
    train_and_test_inadvance_middle_fusion(train_loader, test_loader, model, criterion, optimizer, num_epochs,args.outfile)
