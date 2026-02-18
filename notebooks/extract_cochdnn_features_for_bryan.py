from argparse import ArgumentParser
import sys
sys.path.append('/om/user/jmhicks/projects/TextureSimilarity/code/')
import texture_similarity.utils as ts
import pandas as pd
import numpy as np

architectures = ['kell2018', 'resnet50']
tasks = ['word', 'speaker', 'audioset', 'word_speaker_audioset']
layers = {'kell2018': ['input_after_preproc',
                       'relu0',
                       'relu1',
                       'relu2',
                       'relu3',
                       'relu4',
                       'relufc',],
          'resnet50': ['input_after_preproc',
                       'conv1_relu1',
                       'layer1',
                       'layer2',
                       'layer3',
                       'layer4',
                       'avgpool']}
networks = []
for architecture in architectures:
    for task in tasks:
        for layer in layers[architecture]:
            networks.append({'name': f'{architecture}_{task}',
                            'layer': layer})

def load_sound_list(sound_set):
    sound_path = '/om2/user/bjmedina/BOLIVIA2024/assets/{}/soundlist.csv'
    df = pd.read_csv(sound_path.format(sound_set))
    sound_list = df.stim_path    
    return sound_list

def main():
    parser = ArgumentParser()
    parser.add_argument('--network', type=int, help='network ID; see `network_list`')    
    parser.add_argument('--sound_set', type=str, help='name of stimulus set')    
    args = parser.parse_args()
     
    model_name = networks[args.network]['name']
    layer = networks[args.network]['layer']
    sound_list = load_sound_list(args.sound_set)

    print('Extracting features', flush=True)    
    features = ts.features.extract_from_cochdnn(model_name, layer, sound_list)

    print('Saving features', flush=True)
    filename = f'/om2/user/bjmedina/BOLIVIA2024/assets/{args.sound_set}/{model_name}-{layer}.npy'
    numpy_features = features.cpu().numpy()
    np.save(filename, numpy_features)

if __name__ == "__main__":
    main()