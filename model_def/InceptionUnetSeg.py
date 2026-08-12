import torch, sys
import torch.nn as nn

'''
    Building block for convolutions in discriminators
'''        
class Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1,\
                 padding=1, activate=False, dropout=False, p_drop=0.0):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=kernel_size, \
                                stride=stride,padding=padding, dilation=1)
        self.relu = nn.LeakyReLU(0.2)
        self.BN = nn.BatchNorm3d(cout)
        self.activate = activate
        self.dropout=dropout
        if dropout:
            self.dlayer = nn.Dropout3d(p_drop)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        if self.dropout:
            x = self.dlayer(x)
        if self.activate:
            x = self.relu(x)
        return x

class InceptionV1(nn.Module):
    def __init__(self, cin, c1, c2, c3, c4, activate=True, k1=1, k2=3, k3=5, dropout=False, p_drop=0.0):
        super(InceptionV1, self).__init__()

        self.p1 = Conv(cin, c1, kernel_size=k1, stride=1, padding=k1//2, activate=activate, dropout=dropout, p_drop=p_drop)
        
        self.p2_1 = Conv(cin, c2[0], kernel_size=1, stride=1, padding=0, activate=activate)
        self.p2_2 = Conv(c2[0], c2[1], kernel_size=k2, padding=k2//2, stride=1, activate=activate, dropout=dropout, p_drop=p_drop)

        self.p3_1 = Conv(cin, c3[0], kernel_size=1, stride=1, padding=0, activate=activate)
        self.p3_2 = Conv(c3[0], c3[1], kernel_size=k3, padding=k3//2, stride=1, activate=activate, dropout=dropout, p_drop=p_drop)

        self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv(cin, c4, kernel_size=1, stride =1, padding=0, activate=activate)

    def forward(self, x):
        c1 = self.p1(x)
        c2 = self.p2_2(self.p2_1(x))
        c3 = self.p3_2(self.p3_1(x))
        c4 = self.p4_2(self.p4_1(x))
        return torch.cat((c1,c2,c3,c4), axis=1)

class InUC(nn.Module):
    def __init__(self, cin, n0, activate=False, skip=False, dropout=False, p_drop=0.0):
        super(InUC, self).__init__()
        # assert n0 % 8 == 0
        self.I1 = InceptionV1(cin, n0//4, (n0//4, n0//4), (n0//4, n0//4), n0//4, \
                                activate, 1,3,5, dropout=dropout, p_drop=p_drop)
        self.activate = nn.LeakyReLU(0.2)
        self.MP = nn.AvgPool3d(2)
        self.skip = skip

    def forward(self, x):
        x = self.I1(x)
        # redundant activation
        # x = self.activate(x)
        x = self.MP(x)
        if self.skip:
            return x
        else:
            return x, x.clone()



class InUE(nn.Module):
    '''
        Expansion module
        arguments:
            input channel, conacat channel number, output number
    '''
    def __init__(self, cin, c_cat, n0, activate=False, skip=False, dropout=False, p_drop=0.0):
        super(InUE,self).__init__()
        # assert n0 % 8 == 0
        self.up = nn.ConvTranspose3d(c_cat, c_cat, kernel_size=2, stride=2)
        self.I1 = InceptionV1(c_cat, n0//4, (n0//4, n0//4), (n0//4, n0//4), n0//4, \
                                activate, 1, 3, 5, dropout=dropout, p_drop=p_drop) 
        self.activate = nn.LeakyReLU(0.2)
        self.skip = skip

    def forward(self, x, y=None):
        if not self.skip:
            x = torch.cat([x,y], 1)
        x = self.activate(self.up(x))
        x = self.I1(x)
        # another one; actiavtion occurs in Convs in InceptionV1
        # x = self.activate(x)
        return x


class IncepUnetGen(nn.Module):
    '''
        A U-net using inception blocks
        std U-net:
            conv-conv-reduce
        incep U-net:
            inception-reduce
    '''
    def __init__(self, cin, n_class, n0, noise=False, encode_mass = False,\
                dev='cuda' if torch.cuda.is_available() else 'cpu', \
                skips=False, dropout=False, p_drop=0.0):
        super(IncepUnetGen,self).__init__()
        self.device = torch.device(dev)
        self.noise = noise
        c_bn = 8*n0
        if noise:
            c_bn += 1
        if encode_mass:
            c_bn += 1
        self.encode_mass = encode_mass
        self.skip = skips
        if self.skip:
            self.C1 = InUC(cin, 4*n0, activate=True, skip=True, dropout=dropout, p_drop=p_drop)
        else:            
            self.C1 = InUC(cin, 4*n0, activate=True, dropout=dropout, p_drop=p_drop)
        self.C2 = InUC(4*n0, n0*6, activate=True, dropout=dropout, p_drop=p_drop)
        self.C3 = InUC(n0*6, c_bn, activate=True, dropout=dropout, p_drop=p_drop)

        self.neck = InceptionV1(c_bn, c_bn//2, \
                            (4*n0, c_bn//2), \
                            (4*n0, c_bn//2), c_bn//2, \
                            True, 1, 3, 5, dropout=dropout, p_drop=p_drop)
        
        self.E3 = InUE(2*(c_bn//2), 3*c_bn, n0*6, True, dropout=dropout, p_drop=p_drop)
        self.E2 = InUE(n0*6, n0*12, n0*4, True, dropout=dropout, p_drop=p_drop)
        if self.skip:
            self.E1 = InUE(n0*4, n0*4, n0, True, skip=True, dropout=dropout, p_drop=p_drop)            
        else:
            self.E1 = InUE(n0*4, n0*8, n0, True, dropout=dropout, p_drop=p_drop)
        self.final = nn.Conv3d(n0, n_class, kernel_size=1, stride=1, padding=0)


    def add_noise(self,x):
        '''
            append random noise channel
            to each batch
        '''
        d = x.size()[-1]
        rnd = torch.randn((x.size()[0], 1,d,d,d)).to(self.device)
        return torch.cat((x, rnd), 1)

    def add_mass_channel(self, x, mass):
        '''
            add a channel that encodes the stellar mass
        '''
        d = x.size()[-1]
        masses = torch.ones(x.size()[0],1,d,d,d).to(self.device)
        for i,m in enumerate(mass):
            masses[i] *= m
        return torch.cat((x,masses), 1)

    def forward(self, x, mass = None):

        if self.skip:
            x = self.C1(x)
        else:
            x, cat1 = self.C1(x) # n0, n0
        x, cat2 = self.C2(x) # 2n0, 2n0
        x, cat3 = self.C3(x) # 4n0, 4n0

        
        # if using as generator, 
        # could add noise or mass channels to bottleneck
        if self.noise:
            x = self.add_noise(x)
        if self.encode_mass:
            x = self.add_mass_channel(x, mass)
        x = self.neck(x)    # 8n0
        
        x = self.E3(x, cat3.to(self.device)) # 8n0 + 4n0 -> 4n0
        x = self.E2(x, cat2.to(self.device)) # 
        if self.skip:
            x = self.E1(x)
        else:
            x = self.E1(x, cat1.to(self.device))
        x = self.final(x)
        return x