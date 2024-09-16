import qupath.lib.images.servers.openslide.jna.*
import qupath.lib.projects.ProjectIO

// load OpenSlide
OpenSlideLoader.getLibraryVersion()

if (args.size() == 0)
    println 'No arguments'
else
    println 'Args: ' + args

def imageFolder = args[0]
def geojsonFolder = args[1]
def qupathProject = args[2]

def project = ProjectIO.loadProject(new File(qupathProject), BufferedImage.class)

File[] imageFilePaths = new File(imageFolder).listFiles().findAll{
    x -> x.isFile() &&
    (x.getName().endsWith('.svs') || x.getName().endsWith('.ndpi') || x.getName().endsWith('.mrxs'))
}
String[] imageFilePathsString = imageFilePaths.collect(x -> x.getName())

for (entry in project.getImageList()) {
    def imageName = entry.getImageName()
    if (imageName in imageFilePathsString) {
        println "\nProcessing image $imageName ...\n"

        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()

        def imageNameStem = imageName.take(imageName.lastIndexOf('.'))

        File[] geojsonFilePaths = new File(geojsonFolder).listFiles({
            it.isFile() and it.name.contains("$imageNameStem")
        } as FileFilter)
        // println(geojsonFilePaths)
        for (geojsonFilePath in geojsonFilePaths) {
            // check if the GeoJSON file exists
            if (geojsonFilePath.name.endsWith('.json')) {
                def pathObjects = PathIO.readObjects(geojsonFilePath)
                try {
                    hierarchy.addObjects(pathObjects)
                } catch (Exception e) {
                    println e + imageName
                }
            } else {
                println "GeoJSON file(s) not found for image: $imageNameStem"
            }
        }
        entry.saveImageData(imageData)
    }
}
