# DIL
## Drone Imaging Language

## Installation
DIL has a few dependencies.  It's written in Python, 3.5+ should work.  It also requires
[Hugin](http://hugin.sourceforge.net/) for stitching panoramas.  To install the dependencies on Ubuntu:

    $ sh setup.sh

It will prompt you for superuser rights.  The script is only 5 lines, you can double check it if you like. 

## Usage

  $ python dil.py my_file.dil

## DIL Syntax

DIL files are plain text.  They have the file extension `.dil`.  Currently 5 commands are supported:

    load
    highlight
    stitch
    show
    save

DIL files contain 1 command per line.  Commands can occur in any order, but certain orderings may be nonsensical.
For example, trying to `save` before `load`ing any images won't do anything.

## Command syntax

### load
Load images to be processed.  Can contain more than one filename.  Loading a set of files will clear any previous
loads from memory.  Unsaved changes will be lost, use `save` before loading if you want to retain highlights or
stitched panoramas.

Example:

    load filename_1.jpg filename2.jpg filename3.jpg ...

### highlight
Highlight objects in the loaded image set.  More than one object class may be listed. All
Object detection isn't perfect - results depend on image quality, orientation of the object, and how the
CPU is feeling on a given day.  Supported object classes can be found in `lib/classes.txt`.

Example:

    highlight car bench person

### stitch
Stitch the loaded image set into a 360 panorama.  Doesn't take any parameters.  If you save after using `stitch`, the
image will be saved as `panorama.jpg` and an HTML visualizer will be created at `index.html`.  The visualizer lets you
zoom and pan around the image.

The raw panorama isn't usually good to look at, it is projected in such a way that 360 pano viewers can un-warp it
for human viewing e.g. in a web browser.

Example:

    stitch

### show
Show the panorama created by `stitch`.  It launches your web browser and displays the visualizer.  It doesn't take
any parameters.

If you use `show` without previously `stitching`, you won't get a panorama :)

Example:

    show

### save
Save the current files.  It takes no parameters.  If you have used `highlight`, any detected objects will be saved in
those images.  If you have created a panorama with `stitch`, the raw panorama image will be saved to `panorama.jpg`
and the visualizer demo will be saved to `index.html`.

## Future possibilities

- file globbing
- maybe ocr
- ???
