import zipfile
import pandas as pd

class DailyDialog_Loader():

    def __init__(self, filename):

        self._filename = 'label_data\\' + f'\{filename}'
        self._dialogue_dict = None
        self._act_labels = None

        # Open Zip file
        self._open_zip()

        # Perform mapping
        self._df_file = self._map_utter_act()

    def _get_act_mapping(self) -> dict:
        return {
            '1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'
        }

    def _open_zip(self):

        with zipfile.ZipFile(self._filename) as z:
            
            files = z.namelist()

            # Get the text and label file name
            text_file, label_file = files[3], files[1]

            # Extract the text
            self._dialogue_dict = self._extract_dialog(z, text_file)

            # Extract the labels
            self._act_dict = self._extract_act_labels(z, label_file)

        return None
    
    def _extract_dialog(self, zip, filename) -> dict:

        utterances_dict = {}
        with zip.open(filename, 'r') as f:

            for index, line in enumerate(f.readlines()):

                # For conversion from binary to string format
                line = line.decode('utf-8')

                # Replace the un-processed character
                # Split and remove empty line from the split
                utterances = line.replace('’',"'").split('__eou__')
                utterances.pop(-1)
                utterances_dict[index] = utterances

        return utterances_dict

    def _extract_act_labels(self, zip, filename):
        
        act_dict = {}
        with zip.open(filename, 'r') as f:

            for index, line in enumerate(f.readlines()):
                # For conversion from binary to string format
                line = line.decode('utf-8')
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
            # This has been resolved with Yanran Li
            if len(utterances) != len(value):
                print(f'Mismatch at Index: {int(key) + 1}')
                continue
            
            # Value decremented, as PyTorch counts from 0 to N
            value = [int(val) - 1 for val in value]

            act_list += value
            utter_list += utterances

        data = {
            'dialog_act': act_list,
            'utterance': utter_list
        }

        return pd.DataFrame(data)

    def fetch_dataframe(self, transform_label=False) -> pd.DataFrame:

        if transform_label:
            mapper = self._get_act_mapping()

            # For correct mapping
            self._df_file['dialog_act'] = self._df_file['dialog_act'].map(lambda x: mapper[x+1])

        return self._df_file