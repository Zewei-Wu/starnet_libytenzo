import torch.nn as nn
import torch, sys


class Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1,\
                     activate=True, dropout=True, p_drop=0.0):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=kernel_size, \
                                stride=stride,padding=padding, dilation=1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm3d(cout)
        self.dropout=dropout
        self.drop = nn.Dropout3d(p_drop)
        self.activate = activate

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.drop(x)
        x = self.BN(x)
        if self.activate:
            x = self.relu(x)
        return x

class Inception(nn.Module):
    def __init__(self, cin, c1, c2, c3, c4, drop=False, p_drop=0.2):
        super(Inception, self).__init__()
        # ensure data types
        cin = int(cin)
        c1 = int(c1)
        c2 = [int(c) for c in c2]
        c3 = [int(c) for c in c3]
        c4 = int(c4)
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout3d(p_drop)
        self.p1 = Conv(cin, c1, kernel_size=1, stride=1, padding=0, activate=True)
        
        self.p2_1 = Conv(cin, c2[0], kernel_size=1, stride=1, padding=0, activate=True)
        self.p2_2 = Conv(c2[0], c2[1], kernel_size=3, padding=1, stride=1, activate=True)

        self.p3_1 = Conv(cin, c3[0], kernel_size=1, stride=1, padding=0, activate=True)
        self.p3_2 = Conv(c3[0], c3[1], kernel_size=5, padding=2, stride=1, activate=True)

        self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv(cin, c4, kernel_size=1, stride =1, padding=0, activate=True)

    def forward(self, x):
        c1 = self.p1(x)
        if self.drop:
            c1 = self.dropout(c1)
        c2 = self.p2_2(self.p2_1(x))
        if self.drop:
            c2 = self.dropout(c2)
        c3 = self.p3_2(self.p3_1(x))
        if self.drop:
          c3 = self.dropout(c3)
        c4 = self.p4_2(self.p4_1(x))
        if self.drop:
            c4 = self.dropout(c4)
        return torch.cat((c1,c2,c3,c4), axis=1)

class FirstConvBlock(nn.Module):
    def __init__(self, cin, n0=64, drop=False, p_drop = 0.2):
        super(FirstConvBlock, self).__init__()
        '''
            prepare the input for the inception layers.  Theyre expensive, 
            so we need to reduce the dimensionality!  Design is for 3 channels,
            so cout is scaled by the standard googlenet output here (192 channels)
            * max(cin//3, 1)

        '''
        self.cout = 3*n0
        self.cmid = n0
        if drop:
            self.process = nn.Sequential(\
                                Conv(cin, self.cmid, kernel_size=7, stride=2, padding=3, dropout=True),\
                                nn.MaxPool3d(3, stride=2, padding=1),\
                                Conv(self.cmid, self.cout, kernel_size=3, padding=1, stride=1, dropout=True),\
                                nn.MaxPool3d(3, stride=2, padding=1))
        else:     
            self.process = nn.Sequential(\
                                Conv(cin, self.cmid, kernel_size=7, stride=2, padding=3),\
                                nn.MaxPool3d(3, stride=2, padding=1),\
                                Conv(self.cmid, self.cout, kernel_size=3, padding=1, stride=1),\
                                nn.MaxPool3d(3, stride=2, padding=1))
    def forward(self, x):
        return self.process(x)


class InceptionOne(nn.Module):
    def __init__(self, n0=64, drop = False, p_drop=0.2): 
        super(InceptionOne, self).__init__()
        '''
            series of two inception modules
            input is 3*n0 channels
            output is 7.5*n0 channels
        '''
        f = n0
        self.first_layer = n0
        self.I1 = Inception(3*f, 1*f, (1.5*f, 2*f), (0.25*f, 0.5*f), 0.5*f, drop = drop, p_drop = p_drop) 
        c2 = 4*f
        self.I2 = Inception(c2, 2*f, (2*f, 3*f), (0.5*f, 1.5*f), 1.0*f, drop = drop, p_drop = p_drop) 
        self.pool = nn.MaxPool3d(3,2,padding=1)
    def forward(self, x):
        x = self.I1(x)
        x = self.I2(x)
        x = self.pool(x)
        return x

class SmallInception(nn.Module):
    def __init__(self, cin, n_class, n0=64, \
                    dropout=True, p_drop=0.2):
        super(SmallInception, self).__init__()

        self.preprocess = FirstConvBlock(cin, n0, dropout, p_drop)
        self.I1 = InceptionOne(n0, dropout, p_drop)
        self.pool = nn.AdaptiveAvgPool3d((2,2,2))
        self.linear = nn.Sequential(\
                    nn.Linear(int(2**3*7.5*n0),256),\
                    nn.ReLU(), nn.Linear(256,64),\
                    nn.ReLU(), nn.Linear(64,n_class))


    def forward(self, x):
        x = self.preprocess(x)
        x = self.I1(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x



    
