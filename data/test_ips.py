import requests
import json
apiKey = 'HU8af50ccf0318014312fR0R'

def add_white_list(ip):
    url = "https://h.shanchendaili.com/api.html?action=addWhiteList&appKey={}&ip={}".format(apiKey, ip)
    res = requests.get(url)
    print(res.status_code)
    print(res.text)

def get_proxy():
    url = 'https://h.shanchendaili.com/api.html?action=get_ip&key={}&time=10&count=1&protocol=http&type=json&only=0'.format(apiKey)
    res = requests.get(url)
    print(res.status_code)
    data = json.loads(res.text)['list']
    ip = data[0]['sever']
    port = data[0]['port']
    return '{}:{}'.format(ip, port)


ip = '210.30.107.87'
add_white_list(ip)

proxy = get_proxy()

url = "http://www.a-hospital.com/"

proxies = {
    "http": 'http://{}'.format(proxy),
    "https": 'http://{}' . format(proxy),
}

resp = requests.get(url, proxies=proxies)
print(resp.status_code)
print(resp.text)


