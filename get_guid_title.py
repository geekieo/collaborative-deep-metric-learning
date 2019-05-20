# -*- coding: utf-8 -*-
"""
Created on  2017/12/14 15:15

@author: zhoukang
"""

import requests
import json
import xml.dom.minidom
import cv2
import numpy as np

import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


def get_video_name_use_guid(guid):
    try:
        query_url = r'https://api.3g.ifeng.com/api_phoenixtv_details?guid=%s' % (guid)
        r = requests.get(query_url, timeout=2)
        # result = json.loads(r.content.strip())
        result = json.loads(str(r.content.decode('utf8').strip()))
        return result['singleVideoInfo'][0]['title']
    except Exception as e:
        print(e)
        return ''

def get_video_coverimg_use_guid(guid):
    imgurl = ''
    try:
        query_url = r'https://api.3g.ifeng.com/api_phoenixtv_details?guid=%s' % (guid)
        r = requests.get(query_url, timeout=5)
        # result = json.loads(r.content.strip())
        result = json.loads(str(r.content.decode('utf8').strip()))
        imgurl = result['singleVideoInfo'][0]['smallImgURL']
        r1 = requests.get(imgurl, timeout=2)
        img_array = np.asarray(bytearray(r1.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        return img
    except Exception as e:
        print(e, imgurl)
        return None

simid2title = {}


def get_title_by_simid(simid):
    doctype=''
    if simid in simid2title:
        title = simid2title[simid]
    else:
        try:
            query_url = r'http://10.90.14.19:8082/solr46/item/select?q=simID:%s&fl=title&fl=doctype' % (simid)
            r = requests.get(query_url, timeout=5)
            data = str(r.content.decode('utf8'))
            dom = xml.dom.minidom.parseString(data)
            title = dom.getElementsByTagName('result')[0].getElementsByTagName('doc')[0].getElementsByTagName('str')[0]
            title = title.childNodes[0].data.strip().replace(' ', '')
            doctype = dom.getElementsByTagName('result')[0].getElementsByTagName('doc')[0].getElementsByTagName('str')[1]
            doctype = doctype.childNodes[0].data.strip().replace(' ', '')
        except Exception as e:
            # print(e)
            title = ''
    return title+doctype


def get_guid_title(input_file, output_file, b_guid=False):
    f_save = open(output_file, 'w')
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                f_save.write('\n')
                continue
            if 'top' in line:
                f_save.write(line + '\n')
                continue
            if ' ' not in line:
                guid = line
                if not b_guid:
                    title = get_title_by_simid(guid)
                else:
                    title = get_video_name_use_guid(guid)
                f_save.write(title + '\n')
            else:
                guid, score = line.split(' ')
                if not b_guid:
                    title = get_title_by_simid(guid)
                else:
                    title = get_video_name_use_guid(guid)
                f_save.write((title + ' ' + score + '\n').encode('gbk', 'replace').decode('gbk'))
        f_save.close()


if __name__ == '__main__':
    # get_guid_title(r'./dataset/sample_result_1.txt', r'./dataset/sample_result_title_1.txt', b_guid=False)
    # print(get_title_by_simid('clusterId_22389085'))
    img = get_video_coverimg_use_guid('98aaebae-c716-496d-8370-c4461ecc763c')
    cv2.imshow('src',img)
    cv2.waitKey(0)