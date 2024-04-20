# coding=utf8
# 此段代码有用，但是需要整理寄存器的各个数据
import serial              # pip3 install pyserial
import time
import binascii
import cv2

class Gripper():
    def __init__(self):
        # communicate and initialize the gripper 
        self.ser = serial.Serial(port='/dev/ttyUSB0',
                             baudrate=115200,
                             timeout=1,
                             parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE,
                             bytesize=serial.EIGHTBITS)
        #self.activate_gripper_stream = b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30" #page 78
        self.activate_gripper_stream = b"\x01\x06\x01\x00\x00\xA5\x48\x4D" #完全初始化指令
        self.activate_callback = b"\x01\x03\x02\x00\x00\x01\x85\xB2"   #初始化状态反馈
        
        #self.force_set = b"\x01\x06\x01\x01\x00\x64\xD8\x1D" #100%力量
        self.force_set = b"\x01\x06\x01\x01\x00\x1E\x59\xFE" #30%力量
        self.speed_set = b"\x01\x06\x01\x04\x00\x64\xC8\x1C"  #100%速度
        self.status1 = b"\x01\x06\x01\x03\x01\xF4\x78\x21" 

        time.sleep(0.1)    
    def activate_init(self):
        self.ser.write(self.activate_gripper_stream)
        time.sleep(6)
        print("the grippeer is activated")
        self.ser.write(self.force_set)
        self.ser.write(self.speed_set)
        time.sleep(0.1)    

    def gripper_callback(self):   #返回夹取状态，0运动中 1到达位置 2夹住物体 3物体掉落
        gripper_callback = b"\x01\x03\x02\x01\x00\x01\xD4\x72"
        self.ser.write(gripper_callback)
        back = self.ser.readline()[4]
        return back

    # open the gripper
    def open_gripper(self):
        #self.ser.write(self.read_status)
        self.gripper_action(1000)
        print("the gripper is opening.")
        time.sleep(0.1)

    # close the gripper
    def close_gripper(self):
        self.gripper_action(0)           
        print("the gripper is closed.")
        time.sleep(0.1)

    def get_status(self): #获取返回值，并转为16进制输出
        back = self.ser.readline()
        formatted_data = "b'" + ''.join(fr'\x{byte:02X}' for byte in back) + "'"
        print(formatted_data)
        return formatted_data
    
    def gripper_action(self,x):
        action = hex(x)[2:].upper().zfill(4)
        action_1 = action[:2]
        action_2 = action[2:] 
        #print(action)
        sd = '01 06 01 03 '+str(action_1)+' '+str(action_2)
        # sd = '01 06 01 01 00 64'
        crc = calc_crc(sd.replace(' ', ''))
        #print(crc)
        begin = b"\x01\x06\x01\x03" + bytearray.fromhex(action_1)+bytearray.fromhex(action_2)+bytearray.fromhex(crc[0])+bytearray.fromhex(crc[1])
        # formatted_data = "b'" + ''.join(fr'\x{byte:02X}' for byte in begin) + "'"
        # print(formatted_data)
        self.ser.write(begin)
        loop=True
        # while(loop):
        #     s = self.ser.readline()
        #     # print(s)
        #     if s !=b'':
        #         loop=False
        #     else:
        #         loop=True   
        time.sleep(0.1)
def calc_crc(string):
    data = bytearray.fromhex(string)
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for i in range(8):
            if (crc & 1) != 0:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    hex_crc = hex(((crc & 0xff) << 8) + (crc >> 8)) # 返回十六进制
    crc_0 = crc & 0xff
    crc_1 = crc >> 8
    str_crc_0 = '{:02x}'.format(crc_0).upper()
    str_crc_1 = '{:02x}'.format(crc_1).upper()
    return str_crc_0, str_crc_1 # 返回两部分十六进制字符

if __name__ == "__main__":
    #cv2.namedWindow('image')
    gripper_2f140 = Gripper()
    gripper_2f140.activate_init()
    # gripper_2f140.open_gripper()
    #time.sleep(4)
    # gripper_2f140.gripper_action(940)
    # gripper_2f140.gripper_action(600)
    # gripper_2f140.gripper_action(400)
    # close = 1000
    # while(True):
    #     gripper_2f140.gripper_action(close)
    #     result = gripper_2f140.gripper_callback()
    #     print("result:",result)
    #     close = close-50
    #     if(result == 2):
    #         break


    #print("grasp num:",num)
    #gripper_2f140.close_gripper()
