from torch import nn

def build_model_vgg11(dropout_p):
    # print("[INFO] loading model...")
    vgg11 = nn.Sequential(
                    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),               # 0
                    nn.BatchNorm2d(64),                                     # 1
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 2
                    nn.ReLU(True),                                          # 3
                    
                    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 4
                    nn.BatchNorm2d(128),                                    # 5
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 6
                    nn.ReLU(True),                                          # 7
                    
                    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 8
                    nn.ReLU(True),                                          # 9
                    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 10
                    nn.BatchNorm2d(256),                                    # 11
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 12
                    nn.ReLU(True),                                          # 13
                    
                    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 14
                    nn.ReLU(True),                                          # 15
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 16
                    nn.BatchNorm2d(512),                                    # 17
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 18
                    nn.ReLU(True),                                          # 19
                    
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 20
                    nn.ReLU(True),                                          # 21
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 22
                    nn.BatchNorm2d(512),                                    # 23
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 24
                    nn.ReLU(True),                                          # 25
                    
                    nn.AdaptiveAvgPool2d((7, 7)),                           # 26
                    nn.Flatten(1, -1),                                      # 27
                    nn.Linear(512 * 7 * 7, 4096, True),                     # 28
                    nn.BatchNorm1d(4096),                                   # 29
                    nn.ReLU(True),                                          # 30
                    nn.Dropout(dropout_p, False),                           # 31
                    nn.Linear(4096, 4096, True),                            # 32
                    nn.BatchNorm1d(4096),                                   # 33
                    nn.ReLU(True),                                          # 34
                    nn.Dropout(dropout_p, False),                           # 35
                    nn.Linear(4096, 8, True)                                # 36
                )
    vgg11.apply(init_weights)
    
    return vgg11

def build_model_vgg16(dropout_p, model_path):
    # print("[INFO] loading model...")
    vgg16 = nn.Sequential(
                    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),               # 0
                    nn.ReLU(True),                                          # 1
                    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),              # 2
                    nn.BatchNorm2d(64),                                     # 3
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 4
                    nn.ReLU(True),                                          # 5
                    
                    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 6
                    nn.ReLU(True),                                          # 7
                    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),            # 8
                    nn.BatchNorm2d(128),                                    # 9
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 10
                    nn.ReLU(True),                                          # 11
                    
                    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 12
                    nn.ReLU(True),                                          # 13
                    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 14
                    nn.ReLU(True),                                          # 15
                    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 16
                    nn.BatchNorm2d(256),                                    # 17
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 18
                    nn.ReLU(True),                                          # 19
                    
                    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 20
                    nn.ReLU(True),                                          # 21
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 22
                    nn.ReLU(True),                                          # 23        
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 24
                    nn.BatchNorm2d(512),                                    # 25
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 26
                    nn.ReLU(True),                                          # 27
                    
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 28
                    nn.ReLU(True),                                          # 29
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 30
                    nn.ReLU(True),                                          # 31
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 32
                    nn.BatchNorm2d(512),                                    # 33
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 34
                    nn.ReLU(True),                                          # 35
                    
                    nn.AdaptiveAvgPool2d((7, 7)),                           # 36
                    nn.Flatten(1, -1),                                      # 37
                    nn.Linear(512 * 7 * 7, 4096, True),                     # 38
                    nn.BatchNorm1d(4096),                                   # 39
                    nn.ReLU(True),                                          # 40
                    nn.Dropout(dropout_p, False),                           # 41
                    nn.Linear(4096, 4096, True),                            # 42
                    nn.BatchNorm1d(4096),                                   # 43
                    nn.ReLU(True),                                          # 44
                    nn.Dropout(dropout_p, False),                           # 45
                    nn.Linear(4096, 8, True)                                # 46
                )
    # print("[INFO] loading pretrained model...")
    pretrained_model = torch.load(model_path)
    # print(pretrained_model)
    # print("[INFO] initializing weights...")
    for i, j in zip([0, 2, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32], 
                    [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]):
        vgg16[i].weight.data = pretrained_model.get_parameter(f'features.{j}.weight')
        vgg16[i].bias.data   = pretrained_model.get_parameter(f'features.{j}.bias')
    for i, j in zip([38, 42], 
                    [0 ,  3]):
        vgg16[i].weight.data = pretrained_model.get_parameter(f'classifier.{j}.weight')
        vgg16[i].bias.data   = pretrained_model.get_parameter(f'classifier.{j}.bias')
        
    # print("\n[INFO] model build\n")
    
    return vgg16


def build_model_vgg16_no_batchnorm(dropout_p, model_path):
    # print("[INFO] loading model...")
    vgg16 = nn.Sequential(
                    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),               # 0
                    nn.ReLU(True),                                          # 1
                    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),              # 2
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 4
                    nn.ReLU(True),                                          # 5
                    
                    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 6
                    nn.ReLU(True),                                          # 7
                    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),            # 8
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 9
                    nn.ReLU(True),                                          # 10
                    
                    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 11
                    nn.ReLU(True),                                          # 12
                    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 13
                    nn.ReLU(True),                                          # 14
                    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 15
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 16
                    nn.ReLU(True),                                          # 17
                    
                    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 18
                    nn.ReLU(True),                                          # 19
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 20
                    nn.ReLU(True),                                          # 21        
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 22
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 23
                    nn.ReLU(True),                                          # 24
                    
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 25
                    nn.ReLU(True),                                          # 26
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 27
                    nn.ReLU(True),                                          # 28
                    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 29
                    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 30
                    nn.ReLU(True),                                          # 31
                    
                    nn.AdaptiveAvgPool2d((7, 7)),                           # 32
                    nn.Flatten(1, -1),                                      # 33
                    nn.Linear(512 * 7 * 7, 4096, True),                     # 34
                    nn.ReLU(True),                                          # 35
                    nn.Dropout(dropout_p, False),                           # 36
                    nn.Linear(4096, 4096, True),                            # 37
                    nn.ReLU(True),                                          # 38
                    nn.Dropout(dropout_p, False),                           # 39
                    nn.Linear(4096, 8, True)                                # 40
                )
    vgg16.apply(init_weights)
    return vgg16

def build_model_vgg19(dropout_p, model_path):
    vgg19 = nn.Sequential(
        nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),               # 0
        nn.ReLU(True),                                          # 1
        nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),              # 2
        nn.BatchNorm2d(64),                                     # 3
        nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 4
        nn.ReLU(True),                                          # 5
        
        nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 6
        nn.ReLU(True),                                          # 7
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),            # 8
        nn.BatchNorm2d(128),                                    # 9
        nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 10
        nn.ReLU(True),                                          # 11
        
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 12
        nn.ReLU(True),                                          # 13
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 14
        nn.ReLU(True),                                          # 15
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 16
        nn.ReLU(True),                                          # 17
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 18
        nn.BatchNorm2d(256),                                    # 19
        nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 20
        nn.ReLU(True),                                          # 21
        
        nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 22
        nn.ReLU(True),                                          # 23
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 24
        nn.ReLU(True),                                          # 25
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 26
        nn.ReLU(True),                                          # 27
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 28
        nn.BatchNorm2d(512),                                    # 29
        nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 30
        nn.ReLU(True),                                          # 31
        
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 32
        nn.ReLU(True),                                          # 33
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 34
        nn.ReLU(True),                                          # 35
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 36
        nn.ReLU(True),                                          # 37
        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 38
        nn.BatchNorm2d(512),                                    # 39
        nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 40
        nn.ReLU(True),                                          # 41
        
        nn.AdaptiveAvgPool2d((7, 7)),                           # 42
        
        nn.Flatten(1, -1),                                      # 43
        nn.Linear(512 * 7 * 7, 4096, True),                     # 44
        nn.BatchNorm1d(4096),                                   # 45
        nn.ReLU(True),                                          # 46
        nn.Dropout(dropout_p, False),                 # 47
        nn.Linear(4096, 4096, True),                            # 48
        nn.BatchNorm1d(4096),                                   # 49
        nn.ReLU(True),                                          # 50
        nn.Dropout(dropout_p, False),                 # 51
        nn.Linear(4096, 8, True)            # 52
    )
    # print("[INFO] getting pretrained_model")
    pretrained_model = torch.load(model_path, map_location="cpu")
    # print("[INFO] setting weights and biases")
    for i, j in zip([0, 2, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32, 34, 36, 38], 
            [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]):
        vgg19[i].weight.data = pretrained_model.get_parameter(f'features.{j}.weight')
        vgg19[i].bias.data   = pretrained_model.get_parameter(f'features.{j}.bias')
    for i, j in zip([44, 48],
                    [0 ,  3]):
        vgg19[i].weight.data = pretrained_model.get_parameter(f'classifier.{j}.weight')
        vgg19[i].bias.data   = pretrained_model.get_parameter(f'classifier.{j}.bias')
    # print("[INFO] Model build!")
    return vgg19