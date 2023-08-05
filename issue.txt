ttt(
  (conv): QATDynamicConv2d(
    32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=List[torch.Size([4])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=0 pot=False)
    )
  )
  (conv1): QATDynamicConv2d(
    4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=List[torch.Size([128])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=0 pot=False)
    )
  )
  (bn): DynamicBatchNorm2d(128, eps=2.5e-06, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): QATDynamicConv2d(
    128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=List[torch.Size([16])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=0 pot=False)
    )
  )
  (conv_tranpose): QATDynamicConvTranspose2d(
    16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=1, scale=List[torch.Size([8])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=1 pot=False)
    )
  )
  (pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): QATDynamicLinear(
    in_features=8, out_features=20, bias=True
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=List[torch.Size([20])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=0 pot=False)
    )
  )
  (relu): ReLU()
  (fc1): QATDynamicLinear(
    in_features=20, out_features=10, bias=True
    (weight_fake_quant): DynamicLearnableFakeQuantize(
      fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, scale=List[torch.Size([10])], zero_point=List
      (activation_post_process): DynamicLSQObserver(min_val=List, max_val=List ch_axis=0 pot=False)
    )
  )
  (x_post_act_fake_quantizer): DynamicLearnableFakeQuantize(
    fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=Parameter containing:
    tensor([6.1609], requires_grad=True), zero_point=Parameter containing:
    tensor([0.], requires_grad=True)
    (activation_post_process): DynamicLSQObserver(min_val=0.03504753112792969, max_val=99.90595245361328 ch_axis=-1 pot=False)
  )
  (conv_post_act_fake_quantizer): DynamicLearnableFakeQuantize(
    fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=15, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=Parameter containing:
    tensor([5.0915], requires_grad=True), zero_point=Parameter containing:
    tensor([6.], requires_grad=True)
    (activation_post_process): DynamicLSQObserver(min_val=-30.873188018798828, max_val=14.442919731140137 ch_axis=-1 pot=False)
  )
  (bn_post_act_fake_quantizer): DynamicLearnableFakeQuantize(
    fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=15, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=Parameter containing:
    tensor([0.2090], requires_grad=True), zero_point=Parameter containing:
    tensor([7.], requires_grad=True)
    (activation_post_process): DynamicLSQObserver(min_val=-1.543220043182373, max_val=1.5527533292770386 ch_axis=-1 pot=False)
  )
  (view_post_act_fake_quantizer): DynamicLearnableFakeQuantize(
    fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=15, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=Parameter containing:
    tensor([0.0288], requires_grad=True), zero_point=Parameter containing:
    tensor([2.], requires_grad=True)
    (activation_post_process): DynamicLSQObserver(min_val=-0.04723256826400757, max_val=0.1520591676235199 ch_axis=-1 pot=False)
  )
  (relu_post_act_fake_quantizer): DynamicLearnableFakeQuantize(
    fake_quant_enabled=tensor([0], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, ch_axis=-1, scale=Parameter containing:
    tensor([0.0083], requires_grad=True), zero_point=Parameter containing:
    tensor([0.], requires_grad=True)
    (activation_post_process): DynamicLSQObserver(min_val=0.0, max_val=0.23689578473567963 ch_axis=-1 pot=False)
  )
)

def forward(self, x):
    x_post_act_fake_quantizer = self.x_post_act_fake_quantizer(x);  x = None
    conv = self.conv(x_post_act_fake_quantizer);  x_post_act_fake_quantizer = None
    conv_post_act_fake_quantizer = self.conv_post_act_fake_quantizer(conv);  conv = None
    conv1 = self.conv1(conv_post_act_fake_quantizer);  conv_post_act_fake_quantizer = None
    bn = self.bn(conv1);  conv1 = None
    bn_post_act_fake_quantizer = self.bn_post_act_fake_quantizer(bn);  bn = None
    conv2 = self.conv2(bn_post_act_fake_quantizer);  bn_post_act_fake_quantizer = None
    conv_tranpose = self.conv_tranpose(conv2);  conv2 = None
    pool = self.pool(conv_tranpose)
    getattr_1 = conv_tranpose.shape;  conv_tranpose = None
    getitem = getattr_1[0];  getattr_1 = None
    view = pool.view(getitem, -1);  pool = getitem = None
    view_post_act_fake_quantizer = self.view_post_act_fake_quantizer(view);  view = None
    fc = self.fc(view_post_act_fake_quantizer);  view_post_act_fake_quantizer = None
    relu = self.relu(fc);  fc = None
    relu_post_act_fake_quantizer = self.relu_post_act_fake_quantizer(relu);  relu = None
    fc1 = self.fc1(relu_post_act_fake_quantizer);  relu_post_act_fake_quantizer = None
    return fc1
