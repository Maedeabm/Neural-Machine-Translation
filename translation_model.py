import argparse
import NMT_ETL
import utils
import torch
from neural_attention import CustomAttention
from topk_decode import TopKDecode
from sequence_encoder import SequenceEncoder
from Language import Language
from BeamSearchDecoder import BeamSearch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_model', type=str, help='Attention type: dot, general, concat')
    parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer')
    parser.add_argument('--hidden_size', type=int, help='Size of the hidden layer')
    parser.add_argument('--n_layers', type=int, help='Number of layers in the network')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--language', type=str, help='Language to use')
    parser.add_argument('--input', type=str, help='Input sentence')
    parser.add_argument('--max_len', type=int, help='Maximum length of the output sentence')
    parser.add_argument('--beam_size', type=int, help='Size of the beam for beam search')
    parser.add_argument('--batch_size', type=int, help='Batch size for beam search')
    parser.add_argument('--device', type=str, help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=str, help='Random seed')
    return parser.parse_args()

def load_models(args, input_lang, output_lang):
    encoder = EncoderRNN(input_lang.n_words, args.embedding_size, args.hidden_size, args.n_layers, args.dropout)
    decoder = AttentionDecoderRNN(output_lang.n_words, args.embedding_size, args.hidden_size, args.attn_model, args.n_layers, args.dropout)

    encoder.load_state_dict(torch.load(f'encoder_params_{args.language}'))
    decoder.load_state_dict(torch.load(f'decoder_params_{args.language}'))
    decoder.attention.load_state_dict(torch.load(f'attention_params_{args.language}'))

    return encoder.to(args.device), decoder.to(args.device)

def evaluate(args, input_lang, output_lang, encoder, decoder):
    input_tensor = NMT_ETL.tensor_from_sentence(input_lang, args.input, args.device)
    input_length = input_tensor.size()[0]

    encoder_hidden = encoder.init_hidden(args.device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_context = torch.zeros(1, 1, decoder.hidden_size).to(args.device)
    decoder_hidden = encoder_hidden

    topk_decoder = TopKDecode(decoder, decoder.hidden_size, args.beam_size, output_lang.n_words, Language.sos_token, Language.eos_token, args.device).to(args.device)
    decoder_outputs, _, metadata = topk_decoder(decoder_context, decoder_hidden, encoder_outputs, args.max_len, args.batch_size)

    beam_words = torch.stack(metadata['topk_sequence'], dim=0).squeeze(3).squeeze(1).transpose(0, 1)
    beam_length = metadata['topk_length']
    print_sentences(beam_words, beam_length, 'beam')

    greedy_words, _ = greedy_decode(decoder, decoder_context, decoder_hidden, encoder_outputs, args.max_len, args.device)
    print_sentences(greedy_words, mode='greedy')

def greedy_decode(decoder, decoder_context, decoder_hidden, encoder_outputs, max_len, device):
    decoded_words = []
    decoder_input = torch.LongTensor([[Language.sos_token]]).to(device)

    for di in range(max_len):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi.item()

        if ni == Language.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(ni)

        decoder_input = topi

    return decoded_words, None

def print_sentences(words, lengths=None, mode='greedy'):
    if mode == 'greedy':
        sentence = ' '.join([output_lang.index2word[word] for word in words if word != Language.eos_token])
        print(f'greedy > {sentence}')
    elif mode == 'beam':
        for i, (length, ids) in enumerate(zip(lengths, words.tolist())):
            cur_words = [output_lang.index2word[id] for id in ids[:length]]
            sentence = ' '.join(cur_words)
            print(f'beam {i} > {sentence}')

def main():
    args = parse_arguments()
    utils.verify_language_model_parameters(args.language)
    input_lang, output_lang, pairs = NMT_ETL.load_and_prepare_data(args.language)
    torch.random.manual_seed(args.seed)
    args.device = torch.device(args.device)

    encoder, decoder = load_models(args, input_lang, output_lang)
    evaluate(args, input_lang, output_lang, encoder, decoder)

if __name__ == '__main__':
    main()
