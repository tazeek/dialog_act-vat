import zipfile

class DailyDialog_Loader():

    def __init__(self, filename):

        self._filename = 'label_data\\' + f'\{filename}'

        # Open Zip file
        self._open_zip()

    def _open_zip(self):

        with zipfile.ZipFile(self._filename) as z:
            
            files = z.namelist()

            # Get the text and label file name
            text_file, label_file = files[3], files[1]

            # Extract the text
            print(text_file)
            print(label_file)

            # Extract the labels

        return None
        ...

    def _extract_dialog(self):

        ...

    def _extract_act_labels(self):

        ...