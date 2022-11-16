import zipfile

class DailyDialog_Loader():

    def __init__(self, filename):

        self._filename = 'label_data\\' + f'\{filename}'
        self._dialogue_dict = None
        self._act_labels = None

        # Open Zip file
        self._open_zip()

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