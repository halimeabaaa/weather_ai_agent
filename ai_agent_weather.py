import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1)

# Tools
google_search = GoogleSerperAPIWrapper()
weather_search = OpenWeatherMapAPIWrapper()

# Kullanıcı profili
user_profile = {
    "name": "halime",
    "city": "Malatya",
    "likedMinTemp": -10,
    "likedMaxTemp": 0,
    "liked_conditions": ["karlı"],
    "disliked_conditions": ["güneşli"],
    "disease": ["yok"]
}

# Sistem promptu (agent için)
system_prompt = (
    "Sen çok yardımsever bir asistansın. "
    "Kullanıcının profilini incele ve ona uygun yanıtlar üret. "
    "Weather tool ile hava durumu verilerini al. "
    "Havanın rüzgar, nem, polen, uv, sıcaklık, hissedilen sıcaklık ve görüş mesafesi gibi verilerini değerlendir. "
    "Kullanıcının profili ile hava durumunu kıyasla ve uygunluğunu olasılık olarak hesapla. "
    "Sevdiği bir hava ise 'Bugün harika bir gün' gibi yanıt ver. "
    "Sevmediği bir hava ise 'Hava bugün berbat' gibi yanıt ver. "
    "Kullanıcı hava dışı bir şey sorarsa GoogleSearch ile araştır ve cevapla. "
    "Cevabı bilmiyorsan 'Bilmiyorum' de. "
    "Yanıtını sadece düz metin olarak ver, Markdown veya semboller kullanma."
)

# Tools
tools = [
    Tool(
        name="Weather",
        func=weather_search.run,
        description="Şehir bazlı hava durumu bilgisini verir. Kullanıcıdan şehir almayı unutma."
    ),
    Tool(
        name="GoogleSearch",
        func=google_search.run,
        description="Güncel bilgi gerektiğinde kullan."
    )
]

# Agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)

# Kullanıcı sorgusu
query_prompt = (
    f"Kullanıcı profili:\n"
    f"Ad: {user_profile['name']}\n"
    f"Şehir: {user_profile['city']}\n"
    f"Sevdiği hava: {', '.join(user_profile['liked_conditions'])}\n"
    f"Sevmediği hava: {', '.join(user_profile['disliked_conditions'])}\n"
    f"Hastalık: {', '.join(user_profile['disease'])}\n\n"
    "Lütfen bugünkü hava durumunu açıkla, kullanıcıya uygunluğu değerlendir ve öneriler ver. "
    "Sadece düz metin olarak yanıtla."
)

# Agent’i çalıştır
response = agent_executor.run(query_prompt)
print(response)
