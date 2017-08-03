float angle = 0;
BufferedReader reader;
String line;
PImage img;
PShape face;
int timer = 0;
int fps = 30;

void setup() {
  size(800, 800, P3D);  
  smooth();
  frameRate(fps);
  reader = createReader("quaternion_9dof.txt");
  img = loadImage("face_1.png");
  face = createShape(SPHERE, 150); 
  face.setTexture(img);
}
 

int seconds = 0; 
void draw() {
  background(0);
  lights();
  //if (timer == fps){
  //  seconds++;
  //  timer = 0;
  //}
  timer++;
  seconds = timer / fps;
  textSize(30);
  fill(0, 102, 153);
  
  text(seconds, 10, 30); 
  
  try {
    line = reader.readLine();
  } catch (IOException e) {
    e.printStackTrace();
    line = null;
  }
  if (line == null) {
    // Stop reading because of an error or file is empty
    noLoop();  
  } else {
    String[] pieces = split(line, " ");
    int pitch = int(pieces[0]);
    int roll = int(pieces[1]);
    int yaw = int(pieces[2]);
    
    pushMatrix();
    translate(width/2, height/2);
    //rotateX(roll / 57.3);
    //rotateY(yaw / 57.3);
    //rotateZ(pitch / 57.3);
    
    //rotateX((pitch - 56.0) / 57.3);
    //rotateY((roll - 56.0) / 57.3);
    //rotateZ((yaw - 56.0) / 57.3);
    
    rotateY((yaw) / 57.3);
    rotateX((pitch) / 57.3);
    rotateZ((roll) / 57.3);
    
    noStroke();
    shape(face);
    popMatrix();
  }
}