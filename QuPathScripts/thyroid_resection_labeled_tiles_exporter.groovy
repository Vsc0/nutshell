// Script Editor -> Run for project

import qupath.lib.images.servers.LabeledImageServer

def dirOut = 'OUTPUT_DIRECTORY'
def args = [dirOut, 0.25, 512, 0, 'HP', 'NIFTP', 'PTC', 'Other']
// Each tile of 512 pixels corresponds to a field of view that is 512 * 0.25 = 128 um

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(
    args[0], 'tiles_' + args[1] + '_' + args[2] + '_' + args[3] + '_' + args[4..-1].join('_'), name
)
mkdirs(pathOutput)

double requestedPixelSize = args[1].toDouble()
double averagedPixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
// println 'averagedPixelSize ' + averagedPixelSize
double downSample = requestedPixelSize / averagedPixelSize
// println 'downSample ' + downSample

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
        .downsample(downSample)  // define server resolution at which tiles are exported
        .backgroundLabel(255, ColorTools.WHITE)  // specify background label (usually 0 or 255)
        // if a tile contains a white pixel, it is considered a partially annotated tile
        .addLabel('ROI', 0, ColorTools.BLACK)
        // the output labels, the order matters
        .addLabel(args[4], 1)  // 'HP'
        .addLabel(args[5], 2)  // 'NIFTP'
        .addLabel(args[6], 3)  // 'PTC'
        .addLabel(args[7], 4)  // 'Other'
        .multichannelOutput(false)  // if true, each label is a different channel (required for multiclass probability)
        .build()

// Create an exporter that requests corresponding tiles from the original and labeled image servers
new TileExporter(imageData)
        .downsample(downSample)
        .imageExtension('.png')  // define file extension for original pixels (often .tif, .jpg, .png or .ome.tif)
        .labeledImageExtension('.png')
        .tileSize(args[2].toInteger())  // define size of each tile, in pixels units at the export resolution
        .overlap(args[3].toInteger())  // define overlap, in pixel units at the export resolution
        .labeledServer(labelServer)  // define the labeled image server to use, here the one just built
        .annotatedCentroidTilesOnly(true)  // specify whether tiles without any annotations over the tile centroid should be included
        .annotatedTilesOnly(false)  // if true, only export tiles if there is a (labeled) annotation present
        .includePartialTiles(false)  // specify whether incomplete tiles at image boundaries should be included
        .exportJson(true)  // optionally export a JSON file that includes label information and image/label pairs
        .imageSubDir('images')
        .labeledImageSubDir('masks')
        .writeTiles(pathOutput)  // write tiles to the specified directory

println 'Done with ' + name + ' !'
