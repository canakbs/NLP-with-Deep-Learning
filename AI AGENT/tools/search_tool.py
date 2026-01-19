# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8"
                       )
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
search = DuckDuckGoSearchResults()


if __name__ == "__main__":
    query = "apple hissesi ne kadar"
    
    # Doğru kullanım: önce tool'u oluştur, sonra invoke/run et
    search = DuckDuckGoSearchResults()
    result = search.invoke(query)
    
    # veya alternatif olarak DuckDuckGoSearchRun kullanabilirsiniz:
    # search = DuckDuckGoSearchRun()
    # result = search.invoke(query)
    
    print(f"Sonuç: \n{result}")