import pandas as pd

class LabelLoader:

    def __init__(self):

        # Directory attribute for DailyDialog
        self._data_dir = 'label_data\\'

        # Load dialogues
        # One dialogue: many utterances
        self._dialogue_dict = self._load_dialogues()
        
        # Load act annotations
        self._act_dict = self._load_act_labels()

        # Perform mapping
        self._df_file = self._map_utter_act()
    
    def _get_act_mapping(self) -> dict:
        return {
            '1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'
        }

    def _load_dialogues(self) -> dict:

        utterances_dict = {}
        with open(f'{self._data_dir}\dialogues_text.txt', encoding='utf-8') as f:

            for index, line in enumerate(f.readlines()):

                # Replace the un-processed character
                # Split and remove empty line from the split
                utterances = line.replace('’',"'").split('__eou__')
                utterances.pop(-1)
                utterances_dict[index] = utterances

        return utterances_dict

    def _load_act_labels(self) -> dict:
        
        act_dict = {}

        with open(f'{self._data_dir}\dialogues_act.txt', encoding='utf-8') as f:

            for index, line in enumerate(f.readlines()):
                act_dict[index] = line.split()
            
        return act_dict

    def _map_utter_act(self) -> pd.DataFrame:
        
        # Convert from dict to list
        act_list = []
        utter_list = []

        for key, value in self._act_dict.items():

            utterances = self._dialogue_dict[key]

            # Check mismatch between the lengths
            # 01/08/2022: Index 673 has the mistmatch
            if len(utterances) != len(value):
                print(f'Mismatch at Index: {int(key) + 1}')
                continue

            act_list += value
            utter_list += utterances

        data = {
            'label': act_list,
            'utterance': utter_list
        }

        return pd.DataFrame(data)

    def fetch_dataframe(self, transform_label=False) -> pd.DataFrame:

        if transform_label:
            mapper = self._get_act_mapping()

            self._df_file['label'] = self._df_file['label'].map(lambda x: mapper[x])

        return self._df_file