import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain import hub # LangChain Hub'ı import ediyoruz

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# --- 1. Model ve Araçların (Tools) Kurulumu ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
google_search = GoogleSerperAPIWrapper()
weather_search = OpenWeatherMapAPIWrapper()

tools = [
    Tool(
        name="Weather", # Aracı "Weather" olarak bıraktım, çünkü ajan loglarında bu şekilde kullanıyor.
        func=weather_search.run,
        description="Belirtilen bir şehirdeki mevcut hava durumunu almak için kullanılır. Şehir adını parametre olarak vermelisin."
    ),
    Tool(
        name="GoogleSearch",
        func=google_search.run,
        description="Hava durumu dışındaki güncel olaylar, genel bilgiler veya herhangi bir konuda araştırma yapmak için kullanılır."
    )
]

# --- 2. Kullanıcı Profili ---
user_profile = {
    "name": "Halime",
    "city": "Malatya",
    "likedMinTemp": -10,
    "likedMaxTemp": 0,
    "liked_conditions": ["karlı"],
    "disliked_conditions": ["güneşli"],
    "disease": ["bahar alerjisi"]
}

# --- 3. Gelişmiş Prompt Mühendisliği ---

# LangChain Hub'dan, ReAct ajanları için özel olarak tasarlanmış, test edilmiş bir prompt şablonu çekiyoruz.
# Bu şablon, LLM'e Thought/Action/Observation döngüsünü nasıl kullanacağını açıkça öğretir.
prompt_template = hub.pull("hwchase17/react")

# Kullanıcı profilini ajanın her zaman görebileceği şekilde prompt'un bir parçası yapıyoruz.
user_profile_str = (
    f"Senin için önemli olan kullanıcı profili aşağıdadır:\n"
    f"- Ad: {user_profile['name']}\n"
    f"- Şehir: {user_profile['city']}\n"
    f"- Sevdiği Sıcaklık Aralığı: {user_profile['likedMinTemp']}°C - {user_profile['likedMaxTemp']}°C\n"
    f"- Sevdiği Hava Koşulları: {', '.join(user_profile['liked_conditions'])}\n"
    f"- Sevmediği Hava Koşulları: {', '.join(user_profile['disliked_conditions'])}\n"
    f"- Sağlık Bilgisi: {', '.join(user_profile['disease'])}"
)

# Ajanın sistem talimatlarını ve kurallarını içeren tam metin
# Bu metin, ajana hem görevini hem de nasıl davranması gerektiğini net bir şekilde anlatır.
system_instructions = (
    "Sen çok yardımsever bir asistansın. Kullanıcının profilini incele ve ona uygun yanıtlar üret.\n"
    "Hava durumu verilerini Weather aracı ile al. Havanın rüzgar, nem, sıcaklık gibi verilerini değerlendir.\n"
    "Kullanıcının profili ile hava durumunu kıyasla. Sevdiği bir hava ise 'Bugün harika bir gün' gibi, sevmediği bir hava ise 'Hava bugün berbat' gibi yorumlar yap.\n"
    "Hava durumu dışı bir soru sorulursa GoogleSearch aracını kullan.\n"
    "Yanıtını sadece düz metin olarak ver, Markdown veya semboller kullanma.\n\n"
    "Kullanıcı Profil Bilgileri:\n"
    f"{user_profile_str}"
)
original_react_instructions = prompt_template.template

# Kendi sistem talimatlarımızı, orijinal ReAct talimatlarının başına ekleyerek yeni bir şablon oluşturuyoruz
prompt_template.template = system_instructions + "\n\n" + original_react_instructions

# --- 4. Modern Ajan (Agent) Oluşturma ---

# 'create_react_agent', LLM'i, araçları ve prompt'u birleştirerek karar verme mekanizmasını oluşturur.
agent = create_react_agent(llm, tools, prompt_template)

# AgentExecutor, bu karar mekanizmasını alır ve adım adım çalıştırır.
# handle_parsing_errors=True -> Bu parametre, sizin yaşadığınız hatayı yakalar ve ajanın çökmesini engeller.
# Ajan, bir hata aldığında durumu anlayıp kendini düzeltmeye çalışır. Bu, çok önemlidir!
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Ajanın düşünme sürecini terminalde görmek için
    handle_parsing_errors=True
)

# --- 5. Ajanı Çalıştırma ---

# Eski .run() metodu yerine artık .invoke() kullanılıyor. Bu, LangChain'in standart yöntemidir.
response = agent_executor.invoke({
    "input": (
        "Lütfen bugünkü hava durumunu açıkla, benim profilime göre uygunluğunu değerlendir ve öneriler ver. "
        "Yanıtını sadece düz metin olarak ver."
    )
})

# Yanıt, bir dictionary içinde 'output' anahtarıyla döner.
print("\n--- Nihai Yanıt ---")
print(response['output'])