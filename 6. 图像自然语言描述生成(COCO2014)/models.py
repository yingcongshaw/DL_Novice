import torch
from torch import nn
import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet E
        # Remove linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))                     # self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
    
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)   
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size) 
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)             # features = self.embed(features)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DecoderWithRNN(nn.Module):
    def __init__(self, cfg, encoder_dim=14*14*2048):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithRNN, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']             # 9490
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        ############################################################################
        # To Do: define some layers for decoder with RNN
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init : linear layer to find initial input of LSTMCell
        # self.bn : Batch Normalization for encoder's output
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # 随机丢弃
        self.dropout = nn.Dropout(p = self.dropout)

        # 用于解码的LSTM神经元
        self.decode_step = nn.LSTMCell(self.embed_dim, self.decoder_dim, bias=True)

        # 线性层，用于查找 LSTMCell 的初始输入
        self.init = nn.Linear(self.encoder_dim, self.decoder_dim)

        # 编码器输出的批量归一化
        self.bn = nn.BatchNorm1d(self.decoder_dim, momentum=0.5)

        # 编码器空间 -> 词汇分数
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        
        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.reshape(batch_size, -1)
        vocab_size = self.vocab_size
        
        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        # Initialize LSTM state
        init_input = self.bn(self.init(encoder_out))
        h, c = self.decode_step(init_input)  # (batch_size_t, decoder_dim)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, generate a new word in the decoder with the previous word embedding
        # Your Code Here! 
         
        # shaw
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            preds, h, c = self.one_step(embeddings[:batch_size_t, t, :], h[:batch_size_t], c[:batch_size_t])
            predictions[:batch_size_t, t, :] = preds

        sort_ind = sort_ind.cpu()
        ############################################################################
        return predictions, encoded_captions, decode_lengths, sort_ind
    
    def one_step(self, embeddings, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass 
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return hidden state and cell state: h, c
        # Your Code Here!

        # shaw
        h, c = self.decode_step(embeddings, (h, c))  
        preds = self.fc(self.dropout(h))  

        ############################################################################
        return preds, h, c

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        #################################################################
        # To Do: you need to define some layers for attention module
        # Hint: Firstly, define linear layers to transform encoded tensor
        # and decoder's output tensor to attention dim; Secondly, define
        # attention linear layer to calculate values to be softmax-ed; 
        # Your Code Here!

        # shaw 
        # 图片特征空间 -> 注意力空间
        self.encoder_att = nn.Linear(encoder_dim,attention_dim)
        # RNN 隐藏状态空间 -> 注意力空间
        self.decoder_att = nn.Linear(decoder_dim,attention_dim)
        # 注意力空间 -> 注意力分数
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


        #################################################################
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        #################################################################
        # To Do: Implement the forward pass for attention module
        # Hint: follow the equation 
        # "e = f_att(encoder_out, decoder_hidden)"
        # "alpha = softmax(e)"
        # "z = alpha * encoder_out"
        # Your Code Here!

        # shaw  alphasoftmax (fc (relu(fc(encoder_output) + fc(h))))
        #encoder_att: (batch_size, num_pixels, attention_dim)
        encoder_att = self.encoder_att(encoder_out)
        #decoder_att: (batch_size, attention_dim)
        decoder_att = self.decoder_att(decoder_hidden)
        #att: (batch_size, num_pixels)
        att = self.full_att(self.relu(encoder_att + decoder_att.unsqueeze(1))).squeeze(2)
        # 注意力分数 alpha: (batch_size, num_pixels)
        alpha = self.softmax(att)
        # 加权求和后的图片特征 z: (batch_size, encoder_dim)
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)
        
        #################################################################
        return z, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, cfg, encoder_dim=2048):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        ############################################################################
        # To Do: define some layers for decoder with attention
        # self.attention : Attention layer
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init_h : linear layer to find initial hidden state of LSTMCell
        # self.init_c : linear layer to find initial cell state of LSTMCell
        # self.beta : linear layer to create a sigmoid-activated gate
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!

        # shaw
        # 实例化一个注意力模型 
        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # 随机丢弃
        self.dropout = nn.Dropout(p = self.dropout)

        # 用于解码的LSTM神经元
        self.decode_step = nn.LSTMCell(self.embed_dim +self.encoder_dim, self.decoder_dim, bias=True)

        # 初始状态
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Linear(self.decoder_dim, self.encoder_dim)

        # 编码器空间 -> 词汇分数
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)    # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, decode by attention-weighing the encoder's output based 
        # on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        # Your Code Here!

        # shaw
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            preds, alpha, h, c = self.one_step(embeddings[:batch_size_t, t, :],encoder_out[:batch_size_t], 
                                            h[:batch_size_t], c[:batch_size_t])
            predictions[:batch_size_t,t,:] = preds
            alphas[:batch_size_t,t,:] = alpha
        sort_ind = sort_ind.cpu()
        ############################################################################
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self, embeddings, encoder_out, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        # Your Code Here!
        
        # shaw    
        z, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.beta(h))

        # 缩放
        z = z * gate
        h, c = self.decode_step(torch.cat([embeddings,z],dim =1), (h, c))
        preds = self.fc(self.dropout(h))

        ############################################################################
        return preds, alpha, h, c

class Encoder2(nn.Module):
    """
    Encoder
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder2, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet ResNet-101
        # Remove linear and pool layers
        resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))                     # self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
    
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32) 
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size) 
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)             # features = self.embed(features)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class EncoderWithAdaptive(Encoder):
    """
    Adaptive_Encoder.
    """
    def __init__(self, encoded_image_size=14):
        super(EncoderWithAdaptive, self).__init__(encoded_image_size)
        # self.embed_dim = embed_dim
        # self.decoder_dim = decoder_dim
        self.embed_dim = 512
        self.decoder_dim = 512

        resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(0.5)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.avgpool = nn.AvgPool2d(encoded_image_size)
        self.affine_embed = nn.Linear(2048, self.embed_dim)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)

        a_g = self.avgpool(out)  # (batch_size, 2048, 1, 1)
        a_g = a_g.view(a_g.size(0), -1)   # (batch_size, 2048)

        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        v_g = F.relu(self.affine_embed(a_g))

        return out, v_g

class AdaptiveAttention(Attention):
    """
    Adaptive Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(AdaptiveAttention, self).__init__(encoder_dim, decoder_dim, attention_dim)
        self.sentinel_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.affine_s_t = nn.Linear(decoder_dim, encoder_dim)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, s_t):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :param s_t: sentinel vector, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        att_sentinel = self.full_att(torch.tanh(self.sentinel_att(s_t) + self.decoder_att(decoder_hidden)))
        att = torch.cat([att, att_sentinel], dim=1)

        alpha = self.softmax(att)  # (batch_size, num_pixels + 1)

        # c_hat_t = beta * s_t + （1-beta）* c_t
        attention_weighted_s_t = s_t * alpha[:, -1].unsqueeze(1)
        attention_weighted_s_t = self.affine_s_t(self.dropout(attention_weighted_s_t))
        attention_weighted_encoding = (encoder_out * alpha[:, :-1].unsqueeze(2)).sum(dim=1)\
                                      + attention_weighted_s_t  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAdaptiveAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, cfg, encoder_dim=2048):
    # def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, adaptive_att=False):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAdaptiveAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.decoder_dim = cfg['decoder_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary

        self.adaptive_attention = AdaptiveAttention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network
        self.decode_step_adaptive = nn.LSTMCell(2 * self.embed_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.affine_embed = nn.Linear(self.embed_dim, self.decoder_dim)  # linear layer to transform embeddings
        self.affine_decoder = nn.Linear(self.decoder_dim, self.decoder_dim)  # linear layer to transform decoder's output
        self.fc_encoder = nn.Linear(self.encoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


        self.fc_encoder.bias.data.fill_(0)
        self.fc_encoder.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, v_g, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # Flatten image
        # encoder_out, v_g = encoder_out

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        vocab_size = self.vocab_size
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])


            # g_t = sigmoid(W_x * x_t + W_h * h_(t−1))
            # g_t = self.sigmoid(self.affine_embed(self.dropout(embeddings[:batch_size_t, t, :]))
            #                     + self.affine_decoder(self.dropout(h[:batch_size_t])))    # (batch_size_t, decoder_dim)

            # # s_t = g_t * tanh(c_t)
            # s_t = g_t * torch.tanh(c[:batch_size_t])   # (batch_size_t, decoder_dim)

            # h, c = self.decode_step_adaptive(
            #     torch.cat([embeddings[:batch_size_t, t, :], v_g[:batch_size_t, :]], dim=1),
            #                 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            # attention_weighted_encoding, alpha = self.adaptive_attention(encoder_out[:batch_size_t], h[:batch_size_t], s_t)

            # preds = self.fc(self.dropout(h)) + self.fc_encoder(self.dropout(attention_weighted_encoding))
            # predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha[:, :-1]

            preds, alpha, h, c = self.one_step(v_g[:batch_size_t, :], embeddings[:batch_size_t, t, :],encoder_out[:batch_size_t], 
                                            h[:batch_size_t], c[:batch_size_t])

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha[:, :-1]

        sort_ind = sort_ind.cpu()
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self,v_g, embeddings, encoder_out, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        # Your Code Here!
        
        g_t = self.sigmoid(self.affine_embed(self.dropout(embeddings)) + self.affine_decoder(self.dropout(h)))    # (batch_size_t, decoder_dim)

        # s_t = g_t * tanh(c_t)
        s_t = g_t * torch.tanh(c)   # (batch_size_t, decoder_dim)

        h, c = self.decode_step_adaptive(torch.cat([embeddings, v_g], dim=1),(h, c))                              # (batch_size_t, decoder_dim)
        attention_weighted_encoding, alpha = self.adaptive_attention(encoder_out, h, s_t)

        preds = self.fc(self.dropout(h)) + self.fc_encoder(self.dropout(attention_weighted_encoding))

        ############################################################################
        return preds, alpha, h, c