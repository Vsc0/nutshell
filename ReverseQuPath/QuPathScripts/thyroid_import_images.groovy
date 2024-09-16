import qupath.lib.images.servers.ImageServers
import qupath.lib.projects.ProjectIO

import qupath.lib.images.servers.openslide.jna.*
OpenSlideLoader.getLibraryVersion()

def imageFolder = args[0]
def qupathProject = args[1]

def project = ProjectIO.loadProject(new File(qupathProject), BufferedImage.class)

File[] imageFilePaths = new File(imageFolder).listFiles().findAll(x -> x.isFile() &&
    (x.getName().endsWith('.svs') || x.getName().endsWith('.ndpi')))
    // drag and drop .mrxs images by hand otherwise the import fails!
    // otherwise Project...->Add images->Choose files
for (imageFilePath in imageFilePaths) {
    def imageName = imageFilePath.getName()
    println "\nAdding image $imageName ...\n"
    def imageServer = ImageServers.buildServer(imageFilePath as String)
    def imageEntry = project.addImage(imageServer.getBuilder())
    imageEntry.setImageName(imageName)
}
project.syncChanges()
//getQuPath().refreshProject()
