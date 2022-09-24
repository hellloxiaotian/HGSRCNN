import torch
import torch.nn as nn
import model.ops as ops

'''
class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
'''
        
class  MFCModule(nn.Module):
	def __init__(self,in_channels,out_channels,gropus=1):
		super(MFCModule,self).__init__()
		kernel_size =3
		padding = 1
		features = 64
		features1 = 32
		distill_rate = 0.5
		self.distilled_channels = int(features*distill_rate)
		self.remaining_channels = int(features-self.distilled_channels)
		self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv1_1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv2_1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv3_1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv5_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv6_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv7_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		self.conv8_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		'''
		self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv4_2 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features1,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv5_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
		self.conv6_1 = nn.Sequential(nn.Conv2d(in_channels=2*features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
		'''
		self.ReLU = nn.ReLU(inplace=True)
	def forward(self,input):
		dit1,remain1 = torch.split(input,(self.distilled_channels,self.remaining_channels),dim=1)
		out1_1=self.conv1_1(dit1)
		out1_1_t = self.ReLU(out1_1)
		out2_1=self.conv2_1(out1_1_t)
		out3_1=self.conv3_1(out2_1)
		out1_2=self.conv1_1(remain1)
		out1_2_t = self.ReLU(out1_2)
		out2_2=self.conv2_1(out1_2_t)
		out3_2=self.conv3_1(out2_2)
		#out3 = torch.cat([out1_1,out3_1],dim=1)
		#out3_t = torch.cat([out1_2,out3_2],dim=1)
		out3_t = torch.cat([out3_1,out3_2],dim=1) 
		out3 = self.ReLU(out3_t)
		#out3 = input+out3
		out1_1t = self.conv1_1_1(input)
		out1_2t1 = self.conv2_1_1(out1_1t)
		out1_3t1 = self.conv3_1_1(out1_2t1)
		out1_3t1 = out3+out1_3t1
		out4_1=self.conv4_1(out1_3t1)
		out5_1=self.conv5_1(out4_1)
		out6_1=self.conv6_1(out5_1)
		out7_1=self.conv7_1(out6_1)
		out8_1=self.conv8_1(out7_1)
		out8_1=out8_1+input+out4_1 
		'''
		out1_c = self.conv1_1(input)
		dit1,remain1 = torch.split(out1_c,(self.distilled_channels,self.remaining_channels),dim=1)
		out1_r = self.ReLU(remain1)
		out1_d = self.ReLU(dit1)
		out2_r = self.conv2_1(out1_r)        
		out2_d = self.conv2_2(out1_d)
		out2 = torch.cat([out2_r,out2_d],dim=1)
		out2_r = torch.cat([remain1,out2_r],dim=1)
		out2_d = torch.cat([dit1,out2_d],dim=1)
		out2_1 = out2+out2_r+out2_d
		out2 = self.ReLU(out2_1)
		out3 = self.conv3_1(out2)
		dit3,remain3 = torch.split(out3,(self.distilled_channels,self.remaining_channels),dim=1)
		out3_r = self.ReLU(remain3)
		out3_d = self.ReLU(dit3)
		out4_r = self.conv4_1(out3_r)
		out4_d = self.conv4_2(out3_d)
		out4 = torch.cat([out4_r,out4_d],dim=1)
		out4_r = torch.cat([remain3,out4_r],dim=1)
		out4_d = torch.cat([dit3,out4_d],dim=1)
		out4_1 = out4+out4_r+out4_d
		out4 = self.ReLU(out4_1)
		out5 = self.conv5_1(out4)
		out5_1 = torch.cat([out3,out5],dim=1)
		out5_1 = self.ReLU(out5_1)
		out6_1 = self.conv6_1(out5_1)
		out6_r = input+out6_1
		'''
		return out8_1


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.b1 = MFCModule(features,features)
        self.b2 = MFCModule(features,features)
        self.b3 = MFCModule(features,features)
        self.b4 = MFCModule(features,features)
        self.b5 = MFCModule(features,features)
        self.b6 = MFCModule(features,features)
        self.ReLU=nn.ReLU(inplace=True)
        #self.conv2 = nn.Sequential(nn.Conv2d(in_channels=6*features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x1 = self.conv1_1(x)
        b1 = self.b1(x1)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b5 = b5+b1
        b6 = self.b6(b5)
        b6 = b6+x1
        #b6 = torch.cat([b1,b2,b3,b4,b5,b6],dim=1)
        #b6 = x1+b1+b2+b3+b4+b5+b6
        #x2 = x1+b1+b2+b3+b4+b5+b6
        x2 = self.conv2(b6)
        temp = self.upsample(x2, scale=scale)
        #temp1 = self.upsample(x1, scale=scale)
        #temp = temp+temp1
        #temp2 = self.ReLU(temp)
        out = self.conv3(temp)
        out = self.add_mean(out)
        return out
