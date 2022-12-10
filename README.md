# EasyOCR trainer

[EasyOCR](https://github.com/JaidedAI/EasyOCR) is a python module for extracting text from image. It is a general OCR that can read both natural scene text and dense text in document. We are currently supporting 80+ languages and expanding.

### Training

Fine tunning currently model

1. Download Model for fine tunning : [english_g2](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) -> unzip and move to destination directory all_data
2. Download sample dataset : [en_sample](https://github.com/JaidedAI/EasyOCR/releases/download/v1.4/en_sample.zip) -> unzip and move to destination directory saved_models
3. Add config yaml in directory config_files/name_config.yaml
   I used the filename en_fine_tunning_config.yaml

   ```
   number: '0123456789'
   symbol: "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €"
   lang_char: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
   experiment_name: 'en_sample'
   train_data: 'all_data'
   valid_data: 'all_data/en_sample'
   manualSeed: 1111
   workers: 6
   batch_size: 32 #32
   num_iter: 300000 # you can custom num_iter
   valInterval: 100 # you can custom interval validation
   saved_model: 'saved_models/english_g2.pth' #'saved_models/en_filtered/iter_300000.pth'
   FT: False
   optim: False # default is Adadelta
   lr: 1.
   beta1: 0.9
   rho: 0.95
   eps: 0.00000001
   grad_clip: 5
   #Data processing
   select_data: 'en_sample' # this is dataset folder in train_data
   batch_ratio: '1' 
   total_data_usage_ratio: 1.0
   batch_max_length: 34 
   imgH: 64
   imgW: 600
   rgb: False
   contrast_adjust: False
   sensitive: True
   PAD: True
   contrast_adjust: 0.0
   data_filtering_off: False
   # Model Architecture
   Transformation: 'None'
   FeatureExtraction: 'VGG'
   SequenceModeling: 'BiLSTM'
   Prediction: 'CTC'
   num_fiducial: 20
   input_channel: 1
   output_channel: 256
   hidden_size: 256
   decode: 'greedy'
   new_prediction: False
   freeze_FeatureFxtraction: False
   freeze_SequenceModeling: False
   ```
4. Start training

   ```
   def get_config(file_path):
       with open(file_path, 'r', encoding="utf8") as stream:
           opt = yaml.safe_load(stream)
       opt = AttrDict(opt)
       if opt.lang_char == 'None':
           characters = ''
           for data in opt['select_data'].split('-'):
               csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
               df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
               all_char = ''.join(df['words'])
               characters += ''.join(set(all_char))
           characters = sorted(set(characters))
           opt.character= ''.join(characters)
       else:
           opt.character = opt.number + opt.symbol + opt.lang_char
       os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
       return opt

   opt = get_config("config_files/en_fine_tunning_config.yaml")
   train(opt, amp=False)
   ```

   Log Training

```
training time:  26.636059284210205
[100/300000] Train loss: 0.01502, Valid loss: 0.00408, Elapsed_time: 26.64084
Current_accuracy : 99.320, Current_norm_ED  : 0.9993
Best_accuracy    : 99.320, Best_norm_ED     : 0.9993
--------------------------------------------------------------------------------
Ground Truth              | Prediction                | Confidence Score & T/F
--------------------------------------------------------------------------------
Zsasz @Johnnie Dean_DMSP  | Zsasz @Johnnie Dean_DMSP  | 0.0232	True
Rupp ? OCCUPIED Wmkx      | Rupp ? OCCUPIED Wmkx      | 0.3271	True
--------------------------------------------------------------------------------

```

5. Run model training in easy_ocr

   1. model:  easyocr default saved in ~.EasyOCR, please move your model trained in ~.EasyOCR/model/en_sample.pth
   2. user_network: in directory user_network  there are neural network file and config. Please create a files
   3. user network : filename en_sample.py -> you can custom filename

   ```
   import torch.nn as nn

   class BidirectionalLSTM(nn.Module):

       def __init__(self, input_size, hidden_size, output_size):
           super(BidirectionalLSTM, self).__init__()
           self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
           self.linear = nn.Linear(hidden_size * 2, output_size)

       def forward(self, input):
           """
           input : visual feature [batch_size x T x input_size]
           output : contextual feature [batch_size x T x output_size]
           """
           try: # multi gpu needs this
               self.rnn.flatten_parameters()
           except: # quantization doesn't work with this 
               pass
           recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
           output = self.linear(recurrent)  # batch_size x T x output_size
           return output

   class VGG_FeatureExtractor(nn.Module):

       def __init__(self, input_channel, output_channel=256):
           super(VGG_FeatureExtractor, self).__init__()
           self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                  int(output_channel / 2), output_channel]
           self.ConvNet = nn.Sequential(
               nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
               nn.MaxPool2d(2, 2),
               nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
               nn.MaxPool2d(2, 2),
               nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
               nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
               nn.MaxPool2d((2, 1), (2, 1)),
               nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
               nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
               nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
               nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
               nn.MaxPool2d((2, 1), (2, 1)),
               nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

       def forward(self, input):
           return self.ConvNet(input)

   class Model(nn.Module):

       def __init__(self, input_channel, output_channel, hidden_size, num_class):
           super(Model, self).__init__()
           """ FeatureExtraction """
           self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
           self.FeatureExtraction_output = output_channel
           self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

           """ Sequence modeling"""
           self.SequenceModeling = nn.Sequential(
               BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
               BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
           self.SequenceModeling_output = hidden_size

           """ Prediction """
           self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


       def forward(self, input, text):
           """ Feature extraction stage """
           visual_feature = self.FeatureExtraction(input)
           visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
           visual_feature = visual_feature.squeeze(3)

           """ Sequence modeling stage """
           contextual_feature = self.SequenceModeling(visual_feature)

           """ Prediction stage """
           prediction = self.Prediction(contextual_feature.contiguous())

           return prediction

   ```
   4. config : filename en_sample.yaml -> you can custom filename

      ```
      network_params:
      input_channel: 1
      output_channel: 256
      hidden_size: 256
      imgH: 64
      lang_list:
              - 'en'
      character_list: 0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

      ```
   5. Interface load model

      ```
      import easyocr

      reader = easyocr.Reader(['en'], recog_network='en_sample')
      ```
