import re
import requests

def getReview(url):
    r = re.search(r"i\.(\d+)\.(\d+)", url)
    shop_id, item_id = r[1], r[2]
    ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

    offset = 0
    number = []
    comment = []
    n = 0
    while True:
        data = requests.get(
            ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        i = 1
        for i, rating in enumerate(data["data"]["ratings"], 1):
            if rating["comment"] != '':
                number.append("test_" + str(n+1).zfill(6))
                comment.append("\""+str(rating["comment"]) + str(" " + "\"\n"))
                n = n + 1

        if i % 20:
            break

        offset += 20
    
    # raw = open(file, "r+")
    # contents = raw.read().split("\n")
    # raw.seek(0)                        # <- This is the missing piece
    # raw.truncate()
    # raw.write('New contents\n')

    f = open("data_clean/test.crash", "r+", encoding="utf-8")
    contents = f.read().split("\n")
    f.seek(0)                     
    f.truncate()
    i = 0
    for i in range(n):
        f.write(number[i] + '\n' + comment[i] + '\n')
    f.close()