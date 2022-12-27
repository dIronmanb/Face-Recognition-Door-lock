from os.path import exists
import socketserver
import socket

import serial
import time

# file 전송하는 클래스
class MyTcpHandler(socketserver.BaseRequestHandler):
    def __init__(self):
        super(MyTcpHandler, self).__init__()

    def handle(self):
        data_transferred = 0
        print('[%s] 연결됨' %self.client_address[0])
        filename = self.request.recv(1024)
        filename = filename.decode()

        if not exists(filename):
            raise FileNotFoundError("There is no file or directory")
            return

        print('파일[%s] 전송 시작...' %filename)
        with open(filename, 'rb') as f:
            try:
                data = f.read(1024)
                while data:
                    data_transferred += self.request.send(data)
                    data = f.read(1024)
            except Exception as e:
                print(e)

        print('전송 완료[%s], 전송량[%d]' %(filename, data_transferred))
# Server
def runServer(HOST, PORT):
    print('+++++파일 서버를  시작++++++')
    print('+++++파일 서버를 끝내려면 \'Crtl + C\'를 누르세요.')

    try:
        server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("+++++ 파일 서버를 종료합니다. +++++")
# Client
def getImageFromServer(filename, HOST, PORT):
    data_transferred = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST,PORT))
        sock.sendall(filename.encode())
        data = sock.recv(1024)

        if not data:
            print('파일[%s]: 서버에 존재하지 않거나 전송중 오류발생' %filename)
            return

        with open('download/' + filename, 'wb') as f:
            try:
                while  data:
                    f.write(data)
                    data_transferred += len(data)
                    data = sock.recv(1024)
            except Exception as e:
                print(e)

    print('파일[%s] 전송종료. 전송량 [%d]' %(filename, data_transferred))
# 아두이노 통신
def send_signal(signal, port):

    py_serial = serial.Serial(
        # Window
        port=port,
        # 보드 레이트 (통신 속도)
        baudrate=9600,
    )

    if signal:
        # denied
        commend = "X"
    else:
        # accepted
        commend = "O"

    py_serial.write(commend.encode())
    time.sleep(0.1)
    if py_serial.readable():
        # 들어온 값이 있으면 값을 한 줄 읽음 (BYTE 단위로 받은 상태)
        # BYTE 단위로 받은 response 모습 : b'\xec\x97\x86\xec\x9d\x8c\r\n'
        response = py_serial.readline()
        # 디코딩 후, 출력 (가장 끝의 \n을 없애주기위해 슬라이싱 사용)
        print(response[:len(response) - 1].decode())

    print("Sending message Successfully.")

# TODO:  Server에서 Image -> denied/accepted signal을 다시 Cilent에게 보내주기


if __name__ == "__main__":
    print()

