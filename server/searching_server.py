
import logging
import json
from server.smart_search import SmartSearch
from server.schemas import ContentItemModel
from typing import List
from pathlib import  Path
from datetime import datetime


logging.basicConfig(level=logging.DEBUG, format='%(threadName)s %(message)s')


path_db_index = "server/db-storage/db-index.bin"
path_db_content = "server/db-storage/db-content.json"
local_db_json = "json:///db-content"


class SearchingServer:

    def __init__(self):
        self.smart_search = SmartSearch()
        self.content = self.open_db()
        #self.load_test()


    def open_db(self):
        if Path(path_db_content).exists() and Path(path_db_index).exists():
            logging.info("open_db")
            self.smart_search.open_file(path_db_index)
            fd = open(path_db_content, 'r', encoding='utf-8')
            return json.load(fd)
        return dict()


    def search(self, str_reuest: str) -> List[ContentItemModel]:
        logging.info(f"::search request:{str_reuest}")

        current_datetime = datetime.now()
        f_datetime = current_datetime.strftime("%H:%M %d-%m-%Y")

        content = self.content.get("content", list())
        hostname = "viix.co"
        result = []
        indices, distances = self.smart_search.search(str_reuest.lower(), k=50)

        for id in range(len(indices)):

            result.append(ContentItemModel(
                url     = local_db_json + f"/{ str(indices[id]) }",
                heading = content[indices[id]],
                description = "",
                icon_url = "https://www.google.com/s2/favicons?domain=" + hostname + "&sz=64",
                hostname = hostname,
                hostname_slug = "",
                distance = f"{distances[id]:.8f}",
                img_url  = "url:///img",
                date     = f_datetime,
                tags     = []
            ))
        logging.info(f"::search results.sz={len(indices)}")
        return result


    def page_to_index(self, url: str):
        logging.info(f"page_to_index:{url}")
        #self.smart_search.add_str_to_index(some_text)
        #self.smart_search.write_index(path_db_index)


    def text_to_index(self, txt: str):
        logging.info(f"text_to_index:{txt}")

        if self.smart_search.add_str_to_index(txt.lower()):
            self.content["content"] = self.content.get("content", list())
            self.content["content"].append(txt)

            with open(path_db_content, 'w', encoding='utf-8') as fd:
                json.dump(self.content, fd, ensure_ascii=False, indent=2)

            self.smart_search.write_index(path_db_index)


    def load_test(self):
        test_txts = "server/db-storage/test-allainews.txt"
        if True:
            logging.info(f"load_test load:{test_txts}")

            file_path = Path(test_txts)
            lines = file_path.read_text(encoding="utf-8").splitlines()

            if self.smart_search.add_texts_to_index([line.lower() for line in lines]):
                self.content["content"] = self.content.get("content", list())
                self.content["content"].extend(lines)

                with open(path_db_content, 'w', encoding='utf-8') as fd:
                    json.dump(self.content, fd, ensure_ascii=False, indent=2)

                self.smart_search.write_index(path_db_index)
