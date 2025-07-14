import os
from os import listdir as LD, makedirs as MDs
from os.path import join as PJ, basename as PBN, dirname as PDN, exists as PE
import csv
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from datetime import datetime as DT


def get_all_file_paths(directory):
    file_paths = []  # List to store the file paths
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(PJ(root, file))
    return file_paths


def PRJ_to_DF(d_):
    # Extract paths from PRJ object (in dictionary form) into a DF

    def d_Pa_to_list(data, parent_key=None):
        # Function to extract paths and their packages
        paths = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively process sub-dictionaries
                paths.extend(d_Pa_to_list(value, key if parent_key is None else parent_key))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'path' in item:
                        paths.append((parent_key, str(item['path'])))  # Add package and path
                    elif isinstance(item, list):  # Handle nested lists in 'extra'
                        paths.extend([(parent_key, str(Path(p))) for p in item])
        return paths  # returns list of paths

    extracted_Pa = d_Pa_to_list(d_)  # returned list of paths will be converted to a DF
    df = pd.DataFrame(extracted_Pa, columns=['package', 'directory'])
    return df


def edit_Fi_name(DF, Fi, rename, replace, insert, append):
    # Function to apply series of name edits and return final name column.
    def F_rename():
        return DF[Fi].where(DF[rename].isna(), DF[rename] + '.' + DF[Fi].str.split('.').str[-1])

    DF[Fi] = F_rename()

    def F_replace():
        def replace_row(value, replace_tuple):
            if pd.notna(replace_tuple):  # Skip NaN
                old, new = replace_tuple
                return value.replace(old, new)
            return value  # Return the value unchanged if replace_tuple is NaN

        return DF.apply(lambda row: replace_row(row[Fi], row[replace]), axis=1)

    DF[Fi] = F_replace()

    def F_insert():
        # Inserts tuple (value of a DF column) into value of another column after splitting at "_".
        def insert_row(value, insert_tuple):
            if pd.notna(insert_tuple):  # Skip NaN
                position, part = insert_tuple
                l_value = value.split('_')
                l_value.insert(position, part)
                value = '_'.join(l_value)
                return value
            return value  # Return the value unchanged if insert_tuple is NaN

        return DF.apply(lambda row: insert_row(row[Fi], row[insert]), axis=1)

    DF[Fi] = F_insert()

    def F_append():
        return (
            DF[Fi].str.split('.').str[0]
            + DF[append].fillna('')
            + '.'
            + DF[Fi].str.split('.').str[-1]
        )

    DF[Fi] = F_append()

    return DF[Fi]


def copy_Fi_DF_row(row):
    Pa_Dst = PDN(row['Pa_Dst'])
    MDs(Pa_Dst, exist_ok=True)
    if os.path.exists(row['Pa_Dst']):
        if os.path.getmtime(row['Pa_Src']) > os.path.getmtime(
            row['Pa_Dst']
        ):  # if file is older than destination, skip it.
            shutil.copy2(row['Pa_Src'], row['Pa_Dst'])  # Copy file and preserve metadata
    row['Instruction'] = 'Processed'  # Change instruction, since the file has now been processed
    row['DT_processed'] = DT.now()
    return row  # Return the modified row


def copy_Fo(Pa_Src_Fo, Pa_Dst_Fo, replace=None):
    """
    Copies all contents from a source folder to a destination folder, preserving metadata.
    Shows detailed progress for the files being copied.
    """
    # Get total number of files to be copied
    total_files = sum(len(Fi) for _, _, Fi in os.walk(Pa_Src_Fo))

    # Initialize a tqdm progress bar
    with tqdm(total=total_files, desc=f'Copying from {Pa_Src_Fo} to {Pa_Dst_Fo}') as pbar:
        for Pa_Src, Pa_Fo, Fi in os.walk(Pa_Src_Fo):
            # Calculate relative path to replicate folder structure
            Rel_Path = os.path.relpath(Pa_Src, Pa_Src_Fo)
            Pa_Dst = PJ(Pa_Dst_Fo, Rel_Path)

            # Create destination folder if it doesn't exist
            MDs(Pa_Dst, exist_ok=True)

            for Fi_Src in Fi:
                # Replace logic for the file name
                def replace_Val(Fi_Src):
                    if replace and pd.notna(replace):  # Ensure replace is not None or NaN
                        Old, New = replace
                        return Fi_Src.replace(Old, New)
                    return Fi_Src  # Return unchanged if no replace logic

                Pa_Src_Fi = PJ(Pa_Src, Fi_Src)
                Pa_Dst_Fi = PJ(Pa_Dst, replace_Val(Fi_Src))

                # Check if the destination file exists and compare timestamps
                if os.path.exists(Pa_Dst_Fi):
                    if os.path.getmtime(Pa_Src_Fi) > os.path.getmtime(
                        Pa_Dst_Fi
                    ):  # If source is newer
                        shutil.copy2(Pa_Src_Fi, Pa_Dst_Fi)  # Copy file and preserve metadata
                else:
                    shutil.copy2(Pa_Src_Fi, Pa_Dst_Fi)  # Copy file if it doesn't exist

                pbar.update(1)  # Update the progress bar


def replace_in_file_Fi_line(DF, Pa_Fi_In, Pa_Fi_Out, C_Src, C_Dst):
    # Read the file
    with open(Pa_Fi_In, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Replace values line by line
    for src, dst in zip(DF[C_Src], DF[C_Dst]):
        lines = [
            line.lower().replace(src.lower(), dst) if src.lower() in line.lower() else line
            for line in lines
        ]

    # Save the updated file
    with open(Pa_Fi_Out, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def replace_in_file_Fo_line(DF, Pa_Fi_In, Pa_Fi_Out, C_Src, C_Dst, C_replace='replace'):
    # Read the file
    with open(Pa_Fi_In, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Replace values line by line
    for src, dst, C_replace_val in zip(DF[C_Src], DF[C_Dst], DF[C_replace]):
        lines = [line.replace(src, dst) for line in lines]  # replace directories
        lines = [
            line.lower().replace(C_replace_val[0].lower(), C_replace_val[1])
            if (isinstance(C_replace_val, tuple) and C_replace_val[0].lower() in line.lower())
            else line
            for line in lines
        ]  # replace values in replace column

    # Save the updated file
    with open(Pa_Fi_Out, 'w', encoding='utf-8') as f:
        f.writelines(lines)


# def Fi_name_edit(Fi, l_remove, l_insert):
#     """Edit file nam (Fi)"""

#     name, suffix = Fi.split('.')

#     l_name_parts = name.split('_') # split name into parts (by underscore)

#     for i in l_remove: # remove extra parts
#         print(l_name_parts)
#         print(l_name_parts[i[0]])
#         print(i[1][0])
#         print(i[1][1])
#         l_name_parts[i[0]] = l_name_parts[i[0]][i[1][0]: i[1][1]]

#     for i in l_insert: # insert parts
#         l_name_parts.insert(i[0], i[-1])

#     new_name = "_".join(l_name_parts) + "." + suffix

# return new_name
