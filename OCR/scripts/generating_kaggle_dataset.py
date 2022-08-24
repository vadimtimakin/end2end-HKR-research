import pandas as pd
import shutil
import os
import filetype
from tqdm import tqdm


def folder_unpacker(current_root, new_root, target_type=None):
    """
    Extracts all files of the desired type from a hierarchy of folders of
    indeterminate size.
    Args:
        current_root (str): The source folder from which files are extracted,
        at the top of the hierarchy.
        new_root (str): The folder to which the extracted files will be copied.
        target_type (str or tuple or list):
        Optional, defaults to None. The type of the target files.
        (Example: jpg, pdf, png; without a dot at the beginning);
        if you want to extract files of the diffrent type at the same way,
        pass their types as a tuple or a list;
        if you don't select the type, all files will be extracted.
        To determine the file type, I used the filetype
        package, in particular the 'guess' function with 'mime'.
        To select the file type you need, see how the above function denotes
        types and pass them to the already given function as an argument.
        List of possible types here:
        "https://pypi.org/project/filetype/"
        Or here:
        "https://github.com/t0efL/Dataset-Fixer/blob/master/file_types.txt"
    The function does not perform any conversions to the original folder,
    files are not deleted after copying to a new folder.
    """

    # Checking types of the arguments.
    if type(current_root) != str:
        msg = "current_root must be str, not {0}.".format(type(current_root))
        raise ValueError(msg)
    if type(new_root) != str:
        msg = "new_root must be str, not {0}.".format(type(new_root))
        raise ValueError(msg)
    if target_type and (type(target_type) not in (str, tuple, list)):
        msg = "target_type must be str, "
        msg += "list or tuple, not {0}.".format(type(target_type))
        raise ValueError(msg)

    # Creating new folder if it doesn't exist.
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    # Extracting only certain types.
    if target_type:

        # Working with multiple file types.
        if type(target_type) is not str:
            def flag(x, y):
                file_type = filetype.guess(x)
                # Fighting with NoneType objects.
                if file_type:
                    file_type = file_type.mime
                else:
                    print("Something went frong with {0} file.".format(x))
                # Flag.
                for i in y:
                    if file_type == i:
                        return True
                return False

        # Working with a single file type.
        else:
            def flag(x, y):
                file_type = filetype.guess(x)
                # Fighting with NoneType objects.
                if file_type:
                    file_type = file_type.mime
                else:
                    print("Something went frong with {0} file.".format(x))
                # Flag.
                return file_type == y

    # Extracting all files that are not folders.
    else:
        target_type = True

        def flag(x, y):
            return bool(x or y)

    for root, dirs, files in os.walk(current_root):
        for file in tqdm(files):
            if flag(os.path.join(root, file), target_type):
                # File copying.
                shutil.copy(os.path.join(root, file), new_root)
            else:
                # Extracting files from detected folders.
                if os.path.isdir(os.path.join(root, file)):
                    folder_unpacker(os.path.join(root, file), new_root,
                                    target_type)


folder_unpacker('/home/toefl/K/nto/kaggle_dataset/test', '/home/toefl/K/nto/kaggle_dataset/train')