from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import numpy as np

# 크롬 드라이버 설정
root_path ='C:/Users/User/Desktop/Project/유통/수요예측/우편번호 크롤링/'

chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창을 열지 않음 (백그라운드 실행)
service = Service(root_path+'chromedriver.exe')  # 크롬드라이버 경로 설정

driver = webdriver.Chrome(service=service, options=chrome_options)

# 우편번호 리스트
address_numbers = np.array([37542, 37544, 37559, 37560, 37575, 37581, 37588, 37611, 37632,
       37633, 37645, 37646, 37650, 37655, 37661, 37676, 37684, 37692,
       37724, 37733, 37735, 37741, 37765, 37772, 37775, 37782, 37787,
       37788, 37791, 37793, 37806, 37811, 37815, 37816, 37817, 37823,
       37826, 37830, 37831, 37836, 37841, 37846, 37853, 37867, 37871,
       37886, 37888, 37890, 37899, 37902, 37903, 37917, 37925, 37926,
       37928, 37933, 38178, 37898, 37799, 37819, 37728, 37742, 37769,
       37789, 37746, 38182, 37571, 38039, 38004, 37946, 37938, 37586,
       37592, 37921, 37907, 37660, 37884, 37700, 37706, 37812, 37810,
       37803, 37708, 38179, 37850, 38176, 37732, 38080, 37534, 37538,
       37530, 37539, 37922, 37553, 37606, 37624, 37628, 37687, 37786,
       37714, 37931, 37889, 37818, 37941, 37935, 37869, 38132, 37737,
       37739, 37518, 37694, 37855, 37912, 38002, 37501, 36431, 37522,
       37779, 37545, 37807, 37515, 37528, 37835, 37690, 37591, 37652,
       37751, 37580, 37681, 38854, 37885, 37842, 37723, 37840, 38051,
       38110, 37839, 37577, 37829, 37940, 37827, 37511, 37757, 37849,
       37930, 37653, 37541, 37608, 37616, 37759, 38019, 38131, 37589,
       50246, 36315, 37738, 37651, 37822, 37517, 37529, 37679, 37767,
       38035, 37895, 37502, 37685, 37596, 37743, 37805, 37578, 37562,
       37666, 37740, 37862, 37906, 37828, 37584, 37604, 36421, 37682],
        )
address_dict = {}
try:
    for addr in address_numbers:
        driver.get(f"https://s.search.naver.com/n/csearch/content/eprender.naver?where=nexearch&pkid=252&q={addr}%20%EC%9A%B0%ED%8E%B8%EB%B2%88%ED%98%B8&key=address_kor")
        time.sleep(2)
        addresses = driver.find_elements(By.CSS_SELECTOR, "span.r_addr")
        address_list = [address.text for address in addresses]
        address_dict[addr] = address_list
        print(f"우편번호 {addr}에 해당하는 주소들: {address_list}")
        while True:
            try:
                next_page = driver.find_element(By.LINK_TEXT, "다음 페이지")
                next_page.click()
                time.sleep(2)
                addresses = driver.find_elements(By.CSS_SELECTOR, "span.r_addr")
                address_list = [address.text for address in addresses]
                address_dict[addr].extend(address_list)
            except:
                break

finally:
    driver.quit()



address_dict_str_keys = {str(key): value for key, value in address_dict.items()}
import json
with open(root_path + 'address_dict.json', 'w', encoding='utf-8') as f:
    json.dump(address_dict_str_keys, f, ensure_ascii=False, indent=4)
gyeongbuk_dict = {}
non_gyeongbuk_keys = []
for key, value_list in address_dict.items():
    gyeongbuk_addresses = [addr for addr in value_list if addr.split()[0].endswith('도')]
    if gyeongbuk_addresses:
        gyeongbuk_dict[key] = gyeongbuk_addresses
    else:
        non_gyeongbuk_keys.append(key)
gyeongbuk_dict_str_keys = {str(key): value for key, value in gyeongbuk_dict.items()}
with open(root_path+'gyeongbuk_dict.json', 'w', encoding='utf-8') as f:
    json.dump(gyeongbuk_dict_str_keys, f, ensure_ascii=False, indent=4)




processed_addresses = {}
for key, value_list in gyeongbuk_dict.items():
    unique_addresses = set()
    for addr in value_list:
        cleaned_addr = addr.replace(",", "").strip()
        first_two_words = " ".join(cleaned_addr.split()[:3])
        new_address = f"{first_two_words}".strip()
        unique_addresses.add(new_address)
    processed_addresses[key] = list(unique_addresses)

processed_addresses_str_keys = {str(key): value for key, value in processed_addresses.items()}
with open(root_path+'processed_addresses.json', 'w', encoding='utf-8') as f:
    json.dump(processed_addresses_str_keys, f, ensure_ascii=False, indent=4)



from collections import Counter
common_addresses = {}
for key, value_list in processed_addresses.items():
    if len(value_list) > 1:
        address_words = [addr.split() for addr in value_list]
        if address_words[0][1] == "포항시":
            third_word_counter = Counter([addr[2] for addr in address_words])
            most_common_third_word = third_word_counter.most_common(1)[0][0]
            common_address = f"{address_words[0][0]} {address_words[0][1]} {most_common_third_word}"
        else:
            common_address = f"{address_words[0][0]} {address_words[0][1]}"
        if common_address.split()[-1] == common_address.split()[-2]:
            common_address = " ".join(common_address.split()[:-1])
        common_addresses[key] = [common_address]
    else:
        value_list = value_list[0].split()
        if value_list[-1] == value_list[-2]:
            value_list = value_list[:-1]
        if value_list[1] != "포항시":
            common_addresses[key] = [f"{value_list[0]} {value_list[1]}"]
        else:
            common_addresses[key] = [" ".join(value_list)]
common_addresses_str_keys = {str(key): value for key, value in common_addresses.items()}
with open(root_path + 'final_addresses.json', 'w', encoding='utf-8') as f:
    json.dump(common_addresses_str_keys, f, ensure_ascii=False, indent=4)
