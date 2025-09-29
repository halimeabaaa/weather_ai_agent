import os
import requests # YENİ: API isteği için eklendi
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain import hub
from langchain.memory import ConversationBufferMemory # YENİ: Hafıza için eklendi

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# --- 1. Model ve Araçların (Tools) Kurulumu ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# --- 2. GÜVENLİ API ve YENİ ARAÇ ---
# ÖNEMLİ NOT: Aşağıdaki fonksiyon, sunucu tarafında çalışacak bir API'yi simüle eder.
# Gerçek bir projede bu bir Google Cloud Function veya bir Flask/FastAPI endpoint'i olmalıdır.
# Bu API, Firestore'a bağlanıp veriyi çeker.
def get_user_profile_from_api(username: str) -> dict:
    """
    Belirtilen kullanıcı adı için bir API'den kullanıcı profili verilerini alır.
    Bu fonksiyon, ajanın doğrudan veritabanına erişmesini engeller.
    """
    try:
        # Örnek API URL'si. Kendi API endpoint'inizi buraya yazmalısınız.
        api_url = f"https://sizin-guvenli-api-adresiniz.com/users/{username}"
        # response = requests.get(api_url)
        # response.raise_for_status() # Hata varsa exception fırlat
        # return response.json()

        # --- ŞİMDİLİK API'Yİ SİMÜLE EDİYORUZ ---
        print(f"\n[API Simülasyonu: '{username}' için profil bilgisi alınıyor...]\n")
        if username.lower() == "halime":
            return {
                "name": "Halime",
                "city": "Malatya",
                "likedMinTemp": -10,
                "likedMaxTemp": 0,
                "liked_conditions": ["karlı"],
                "disliked_conditions": ["güneşli"],
                "disease": ["bahar alerjisi"]
            }
        else:
            return {"error": "Kullanıcı bulunamadı."}
        # --- SİMÜLASYON SONU ---

    except requests.exceptions.RequestException as e:
        return {"error": f"API'ye bağlanırken hata oluştu: {e}"}


# Araçları tanımla
google_search = GoogleSerperAPIWrapper()
weather_search = OpenWeatherMapAPIWrapper()

tools = [
    Tool(
        name="Weather",
        func=weather_search.run,
        description="Belirtilen bir şehirdeki mevcut hava durumunu almak için kullanılır. Şehir adını parametre olarak vermelisin."
    ),
    Tool(
        name="GoogleSearch",
        func=google_search.run,
        description="Hava durumu ve kullanıcı profili dışındaki güncel olaylar, genel bilgiler veya herhangi bir konuda araştırma yapmak için kullanılır."
    ),
    # YENİ: Kullanıcı profilini getiren araç
    Tool(
        name="GetUserProfile",
        func=get_user_profile_from_api,
        description="Belirtilen bir kullanıcı adına ait profil bilgilerini (şehir, sıcaklık tercihleri, sağlık durumu vb.) almak için kullanılır. Kullanıcı adını parametre olarak vermelisin."
    )
]

# --- 3. Gelişmiş Prompt Mühendisliği ---

prompt_template = hub.pull("hwchase17/react")

# DEĞİŞTİ: Sistem talimatları artık dinamik olarak kullanıcı profiliyle doldurulmuyor.
# Ajanın kendisinin profili alması ve anlaması gerekiyor.
system_instructions = (
    "Sen çok yardımsever bir asistansın. Görevin, kullanıcının talebine göre onun profilini bularak hava durumunu yorumlamaktır.\n"
    "1. Adım: Kullanıcının kim olduğunu anla ve 'GetUserProfile' aracını kullanarak profil bilgilerini al.\n"
    "2. Adım: Profildeki şehir bilgisine göre 'Weather' aracını kullanarak mevcut hava durumu verilerini al.\n"
    "3. Adım: Hava durumu verilerini (sıcaklık, nem, rüzgar, koşullar) kullanıcının profiliyle (sevdiği/sevmediği sıcaklıklar, koşullar, sağlık bilgileri) karşılaştır.\n"
    "4. Adım: Bu karşılaştırmaya dayanarak, havanın kullanıcı için ne kadar uygun olduğuna dair **yüzdesel bir olasılık (%0 ile %100 arasında) hesapla.**\n"
    "   - Örnek: 'Bugünkü hava sıcaklığı tam sevdiğin aralıkta ve kar yağıyor, bu yüzden senin için uygunluk oranı %95.'\n"
    "   - Örnek: 'Hava güneşli olduğu için bu durum pek hoşuna gitmeyebilir. Sıcaklık da sevdiğin aralığın biraz üstünde. Uygunluk oranı %30.'\n"
    "5. Adım: Bu olasılığa dayanarak kullanıcıya önerilerde bulun. Örneğin, alerjisi varsa polen durumu hakkında GoogleSearch ile araştırma yapabilirsin.\n"
    "6. Adım: Yanıtını sadece düz metin olarak ver, Markdown veya semboller kullanma.\n"
    "7. Adım: Sohbet geçmişini ('chat_history') dikkate alarak önceki konuşmaları hatırla ve tutarlı cevaplar ver."
)

original_react_instructions = prompt_template.template
prompt_template.template = system_instructions + "\n\n" + original_react_instructions


# --- 4. Modern Ajan (Agent) ve Hafıza Oluşturma ---

# YENİ: Sohbet geçmişi için hafıza nesnesi oluştur
memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(llm, tools, prompt_template)

# DEĞİŞTİ: AgentExecutor'a hafıza eklendi.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory, # YENİ
    verbose=True,
    handle_parsing_errors=True
)


# --- 5. Ajanı Döngü İçinde Çalıştırma ---

print("Yardımcı Asistan'a hoş geldiniz! Çıkmak için 'exit' veya 'quit' yazın.")

while True:
    user_input = input("Siz: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Görüşmek üzere!")
        break

    # Ajana girdi verirken artık sohbet geçmişini de dahil ediyoruz.
    # AgentExecutor bunu `memory` parametresi sayesinde otomatik yönetir.
    response = agent_executor.invoke({
        "input": user_input
    })

    print("\n--- Asistanın Yanıtı ---")
    print(response['output'])
    print("-" * 25)