#include <FastLED.h>
#include <Servo.h> 

Servo Servo1; 

#define servoPin 2
#define LED_COUNT 12 // число пикселей в ленте
#define LED_DT 3    // пин, куда подключен DIN ленты (номера пинов ESP8266 совпадает с Arduino)  

unsigned long actTime = 0;
unsigned long remTime = 0;
const unsigned long period = 400;
bool isLedOn = false;

bool isOpen = false;
bool closingScheduled = false;
unsigned long openedAtMillis = 0;
int openStatePeriod = 10000;
int openedPosition = 20;
int closedPosition = 100;

bool newData = false; // booleans for new data from serial, and runallowed flag
char receivedCommand; //a letter sent from the terminal

char blinkEndColor = 'F';
int blinkCountsRequired = 3;
int blinkCountsPassed = 0;

int pos = 0; // Define the starting position of the servo
int target_pos = 0; // Define the target position of the servo

CRGBArray<LED_COUNT> leds;

void setup(){
  Servo1.attach(servoPin); 
  LEDS.addLeds<WS2811, LED_DT, GRB>(leds, LED_COUNT);  // настройки для вашей ленты (ленты на WS2811, WS2812, WS2812B)
  Serial.begin(9600);
  closePosit();
}

void loop(){
  checkSerial();
  actTime = millis();
  if(blinkCountsPassed != blinkCountsRequired){
      blinkLed(blinkEndColor);
  }else{
    if(pos != target_pos){
        Serial.println("Moving to target");
        Servo1.write(target_pos);    
        pos = target_pos;  
        if(pos == openedPosition){
          openedAtMillis = actTime;  
          Serial.println("Opened"); 
          Serial.print("Current time "); 
          Serial.println(actTime);
          Serial.print("Opened time ");
          Serial.println(openedAtMillis);
          isOpen = true;
          
        }else if(pos == closedPosition){
          Serial.println("Closed"); 
          isOpen = false;         
        }else{
          Serial.println("Unknown position"); 
          isOpen = true;          
        }
    }
  }
  
  if(isOpen && !closingScheduled && (actTime - openedAtMillis > openStatePeriod)){
    Serial.println("Force closing due to timeout");
          Serial.print("Current time "); 
          Serial.println(actTime);
          Serial.print("Opened time ");
          Serial.println(openedAtMillis);
    closingScheduled = true;
    closePosit();
  }
}


void checkSerial() //function for receiving the commands
{	
	if (Serial.available() > 0) //if something comes from the computer
	{
		receivedCommand = Serial.read(); // pass the value to the receivedCommad variable
		newData = true; //indicate that there is a new data by setting this bool to true
		if (newData == true) //we only enter this long switch-case statement if there is a new command from the computer
		{
			switch (receivedCommand) //we check what is the command
			{  
				case 'C':
          if(isOpen){
            Serial.println("Attepmting to close.");
            closePosit();
          }else{
            Serial.println("Already closed"); 
          }
				break;
				
				case 'O':
          if(!isOpen){
            Serial.println("Attepmting to open.");   
            openPosit();
          }else{
            Serial.println("Already opened"); 
          }
				break;
       
				default: break;
			}
		}
		newData = false;		
	}
}


void openPosit(){
  isOpen = false;
  blinkCountsPassed = 0;
  closingScheduled = false;
  blinkEndColor = 'G';
  target_pos = openedPosition;
  pos = 0;
  Serial.println("Opening.");   
  //Serial.println("Opened."); //print action
}
void closePosit(){
  isOpen = true;
  closingScheduled = true;
  blinkCountsPassed = 0;
  blinkEndColor = 'R';
  target_pos = closedPosition;
  pos = 0;
  Serial.println("Closing.");   
  //Serial.println("Closed"); //print action
}
void blinkLed(char endColor){
    if (actTime - remTime >= period){
      remTime = actTime;
      
      if(!isLedOn) {
          isLedOn = true;
          fill_solid(leds, LED_COUNT, CRGB::Yellow);
          //Serial.println("Blinking on");
          blinkCountsPassed+=1;
      } else {
          isLedOn = false;
          fill_solid(leds, LED_COUNT, CRGB::Black);
          //Serial.println("Blinking off");
      }
    }
    FastLED.show();
    if(blinkCountsPassed == blinkCountsRequired){
      setLEDState(endColor);
    }
}


void setLEDState(char color){
  switch(color){
    case 'R':
      fill_solid( leds, LED_COUNT, CRGB::Red);
      Serial.println("Setting LEDs to RED");
      //leds[0] = CRGB(255, 0, 0); // RED
      break;
    case 'G':
      fill_solid( leds, LED_COUNT, CRGB::Green);
      Serial.println("Setting LEDs to GREEN");
      //leds[0] = CRGB(0, 255, 0); // GREEN
      break;
    case 'Y':
      fill_solid( leds, LED_COUNT, CRGB::Yellow);
      Serial.println("Setting LEDs to YELLOW");
      //leds[0] = CRGB(255, 255, 0); // YELLOW
      break;
    case 'F':
      fill_solid( leds, LED_COUNT, CRGB::Black);
      Serial.println("Setting LEDs to off");
      //leds[0] = CRGB(0, 0, 0); // YELLOW
      break;
  }   
  FastLED.show();
}

