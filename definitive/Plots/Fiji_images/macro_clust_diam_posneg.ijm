run("Set Measurements...", "area mean shape feret's redirect=None decimal=3");

run("Options...", "iterations=1 count=1 black");
run("Close All");

InputFolder = getDirectory("Select input control directory"); 
OutputFolder = getDirectory("Select OUTPUT folder to save results");
Files = getFileList(InputFolder);
for(i=0;i<lengthOf(Files);i++)
{
	ImagePath = Files[i];
	open(InputFolder+ImagePath);
	rename("Input");
	setOption("BlackBackground", true);
	run("Mean...", "radius=15");
	run("Convert to Mask");
	
	run("Analyze Particles...", "display exclude clear add");
	imageName = ImagePath;
	saved = "results.csv";
	saved2 = "img.png";
	PathName = OutputFolder + imageName + saved;
	PathName2 = OutputFolder + imageName + saved2;
	saveAs("Results", PathName);
	selectImage("Input");
	saveAs("PNG", PathName2);
	run("Close");
}
