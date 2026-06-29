# What To Do When Professor Sends New Images

This is the simple version.

## Step 1: Put The Data On Desktop

If professor sends a ZIP file:

1. Download it.
2. Double-click it to unzip.
3. Move the unzipped folder to Desktop.

If professor sends a normal folder, just put that folder on Desktop.

The image names should look something like:

```text
case01_before.tif
case01_after.tif
case02_before.tif
case02_after.tif
```

Two images make one tornado-path result. One hundred images usually make around fifty results.

## Step 2: Open The Project Folder

Open:

```text
/Users/m.mahin/Documents/New project 2/Spatial-Analysis
```

## Step 3: Double-Click The Runner

Double-click:

```text
RUN_NEW_DATASET.command
```

Terminal will open.

## Step 4: Drag The Professor Folder Into Terminal

When Terminal asks for the dataset folder:

1. Drag the professor's folder from Desktop into the Terminal window.
2. Press Enter.

Do not open the `.joblib` model file. The script uses it automatically.

## Step 5: Open The Output

When the script finishes, it opens the result folder.

Open this first:

```text
batch_contact_sheet.png
```

That is the quick overview of all predicted tornado paths.

For one specific case, open that case folder and view:

```text
showcase_prediction_map.png
```

## What The Colors Mean

- Red = model predicted tornado damage/path
- Cyan/blue = official NWS path if a shapefile was provided
- Gray/white = unreadable or missing imagery

## If It Does Not Find Pairs

Rename the files so the matching images have the same case name and clear before/after words.

Example:

```text
houston_before.tif
houston_after.tif
```

Not good:

```text
image1.tif
image2.tif
```

The model needs to know which image is before and which image is after.
