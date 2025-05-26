


from DrissionPage import ChromiumPage
import time
from urllib.parse import quote
import random

def sign_in():
    sign_in_page = ChromiumPage()
    sign_in_page.get('https://www.xiaohongshu.com')
    print("请扫码登录")
    # 第一次运行需要扫码登录
    time.sleep(20)

def search(keyword):
    global page
    page = ChromiumPage()
    page.get(f'https://www.xiaohongshu.com/search_result?keyword={keyword}&source=web_search_result_notes')

times = 20
def craw(times):
    for i in tqdm(range(1, times + 1)):
        get_info()
        page_scroll_down()


if __name__ == '__main__':
    # contents列表用来存放所有爬取到的信息
    contents = []

    # 搜索关键词
    keyword = "繁花"
    # 设置向下翻页爬取次数
    times = 20

    # 第1次运行需要登录，后面不用登录，可以注释掉
    # sign_in()

    # 关键词转为 url 编码
    keyword_temp_code = quote(keyword.encode('utf-8'))
    keyword_encode = quote(keyword_temp_code.encode('gb2312'))

    # 根据关键词搜索小红书文章
    search(keyword_encode)

    # 根据设置的次数，开始爬取数据
    craw(times)

    # 爬到的数据保存到本地excel文件
    save_to_excel(contents)