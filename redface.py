import requests

# https://www.xiaohongshu.com/explore/67ee0d99000000001c01f6cd?xsec_token=AB55OXnmze4TGfAwx-zx9yFdFynUUA-EeqyenIEu28kYQ=&xsec_source=pc_cfeed&source=404

# 请求地址
url = 'https://edith.xiaohongshu.com/api/sns/web/v1/feed'

# 请求头
h1 = {
	'Accept': 'application/json, text/plain, */*',
	'Accept-Encoding': 'gzip, deflate, br',
	'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
	'Content-Type': 'application/json;charset=UTF-8',
	'Cookie': 'abRequestId=fc6ba8c5-bbde-5edf-9ef2-570f0b1c6cf8; webBuild=4.62.1; xsecappid=xhs-pc-web; a1=196098d15cb3xv5pnq1wqexlx98a4tb2o9yu53moa50000406419; webId=fb2091099116488ce2b28dc87a681ef8; acw_tc=0a4a190217439169653292020e20e5412c86155e2548e5638849206285531d; websectiga=2a3d3ea002e7d92b5c9743590ebd24010cf3710ff3af8029153751e41a6af4a3; sec_poison_id=ffb749de-270a-48e2-b6b3-1b46ab8ce46c; gid=yjK8jYfqqjljyjK8jYfy2jq6SDqCh2UIAyEAvl0dC1CjY728049DJ288848K4yj8qKq288Wd; web_session=040069b2ae589dcb47badd47c7354b574d4581; loadts=1743917010808',
	'Origin': 'https://www.xiaohongshu.com',
	'Referer': 'https://www.xiaohongshu.com/',
	'Sec-Ch-Ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
	'Sec-Ch-Ua-Mobile': '?0',
	'Sec-Ch-Ua-Platform': '"macOS"',
	'Sec-Fetch-Dest': 'empty',
	'Sec-Fetch-Mode': 'cors',
	'Sec-Fetch-Site': 'same-site',
	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
}


# 请求参数
post_data = {
	"source_note_id": 'https://www.xiaohongshu.com/explore/67b338ad00000000180131f1',
	"image_formats": ["jpg", "webp", "avif"],
	"extra": {"need_body_topic": "1"}
}


# 发送请求
r = requests.post(url, headers=h1, data=post_data)
# 接收数据
json_data = r.json()

print(json_data)
# 笔记标题
try:
	title = json_data['data']['items'][0]['note_card']['title']
except:
	title = ''
# 返回数据
print(title)

# data_row = note_id, title, desc, create_time, update_time, ip_location, like_count, collected_count, comment_count, share_count, nickname, user_id, user_url
# # 保存到csv文件
# with open(self.result_file, 'a+', encoding='utf_8_sig', newline='') as f:
# 	writer = csv.writer(f)
# 	writer.writerow(data_row)
