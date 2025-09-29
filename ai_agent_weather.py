import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain import hub

# .env dosyasındaki ortam değişkenlerini yükle
# Lütfen .env dosyanızda GOOGLE_API_KEY, SERPER_API_KEY ve OPENWEATHERMAP_API_KEY anahtarlarınızın olduğundan emin olun.
load_dotenv()

# --- 1. Model ve Araçların (Tools) Kurulumu ---

# LLM (Büyük Dil Modeli)
# Temperature parametresi modelin ne kadar "yaratıcı" olacağını belirler.
# 0.0 daha deterministik ve tutarlı yanıtlar verir.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Araçlar
# Araçların tanımlarını (description) daha açıklayıcı yapmak, ajanın hangi aracı ne zaman kullanacağını daha iyi anlamasına yardımcı olur.
google_search = GoogleSerperAPIWrapper()
weather_search = OpenWeatherMapAPIWrapper()

tools = [
    Tool(
        name="WeatherSearch",
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

# Kullanıcı profili, ajanın kişiselleştirilmiş yanıtlar üretmesi için kullanılır.
# Bu bilgiyi bir veritabanından veya bir kullanıcı oturumundan da alabilirsiniz.
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

# LangChain Hub'dan hazır bir ReAct (Reasoning and Acting) prompt'u çekiyoruz.
# Bu, ajanın düşünme sürecini (Thought), aksiyonunu (Action) ve gözlemini (Observation) adım adım izlemesini sağlar.
# Bu prompt, ajanın daha güvenilir ve şeffaf kararlar almasına olanak tanır.
prompt_template = hub.pull("hwchase17/react")

# Prompt'un içindeki 'input_variables'a yeni değişkenler ekleyerek prompt'u kişiselleştiriyoruz.
# Bu sayede kullanıcı profili gibi ek bilgileri ajanın temel talimatlarına enjekte edebiliriz.
prompt_template.input_variables.extend(["user_profile"])

# Kullanıcı profilini formatlayarak ajanın anlayacağı bir metin bloğuna dönüştürüyoruz.
user_profile_str = (
    f"İşte sana özel bilgilerim:\n"
    f"- Adım: {user_profile['name']}\n"
    f"- Yaşadığım Şehir: {user_profile['city']}\n"
    f"- Sevdiğim Sıcaklık Aralığı: {user_profile['likedMinTemp']}°C ile {user_profile['likedMaxTemp']}°C arası.\n"
    f"- Sevdiğim Hava Durumları: {', '.join(user_profile['liked_conditions'])}\n"
    f"- Sevmediğim Hava Durumları: {', '.join(user_profile['disliked_conditions'])}\n"
    f"- Sağlık Durumum: {', '.join(user_profile['disease'])} (Bu bilgiye göre polen, nem gibi faktörlere dikkat et.)"
)

# Prompt'un içeriğini güncelliyoruz. Artık her sorguda kullanıcı profilini tekrar tekrar göndermeye gerek kalmayacak.
prompt_template.template = (
    "Sen, kullanıcı profiline göre kişiselleştirilmiş, yardımsever bir asistansın.\n"
    "Sana verilen araçları kullanarak soruları yanıtla. Yanıtlarını aşağıdaki formata uygun olarak vermelisin:\n\n"
    "Question: Yanıtlaman gereken soru.\n"
    "Thought: Ne yapman gerektiğini her zaman düşünmelisin. Kullanıcının şehri profilinde belirtilmiş, bu bilgiyi kullanabilirsin.\n"
    "Action: Kullanacağın aracın adı, `WeatherSearch` veya `GoogleSearch` olabilir.\n"
    "Action Input: Araca vereceğin girdi.\n"
    "Observation: Aracın sana döndürdüğü sonuç.\n"
    "... (Bu Thought/Action/Action Input/Observation döngüsü, nihai yanıta ulaşana kadar devam edebilir)\n"
    "Thought: Artık nihai yanıtı biliyorum.\n"
    "Final Answer: Sorunun orijinal dilinde, kullanıcı profiline göre kişiselleştirilmiş nihai yanıt.\n\n"
    "--- KULLANICI PROFİLİ ---\n"
    f"{user_profile_str}\n"
    "-------------------------\n\n"
    "Şimdi başla!\n\n"
    "Question: {input}\n"
    "Thought: {agent_scratchpad}"
)

# --- 4. Modern Ajan (Agent) Oluşturma ---

# 'initialize_agent' eski bir yöntemdir. Artık 'create_react_agent' gibi daha modern ve esnek fonksiyonlar kullanılıyor.
# Bu fonksiyon, bir LLM ve araçlarla birlikte çalışacak bir ReAct ajanı oluşturur.
agent = create_react_agent(llm, tools, prompt_template)

# AgentExecutor, ajanın çalışma döngüsünü yönetir. Ajanın karar alma, araç kullanma ve gözlem yapma süreçlerini yürütür.
# 'verbose=True' parametresi, ajanın düşünme sürecini terminalde görmemizi sağlar, bu da hata ayıklama için çok faydalıdır.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True, # Olası parse hatalarını yönetir
    max_iterations=5 # Ajanın sonsuz döngüye girmesini engeller
)

# --- 5. Ajanı Çalıştırma ---

def run_assistant(query):
    """
    Kullanıcı sorgusunu alır ve ajanı çalıştırır.
    """
    print(f"\n--- Sorgu Başlatılıyor ---")
    print(f"Sorgu: {query}")
    try:
        # Ajanı, kullanıcı profili ve sorgu ile birlikte çalıştırıyoruz.
        response = agent_executor.invoke({
            "input": query,
            "user_profile": user_profile_str # Prompt'a enjekte ettiğimiz değişken
        })
        print("\n--- Ajanın Yanıtı ---")
        # Yanıtı 'output' anahtarından alıp yazdırıyoruz.
        print(response['output'])
    except Exception as e:
        print(f"\n--- Hata ---")
        print(f"Ajan çalışırken bir hata oluştu: {e}")

if __name__ == "__main__":
    # Kullanıcı için bugünün hava durumunu soran varsayılan sorgu
    default_query = (
        "Bugün hava nasıl? Hava durumunu benim profilime göre değerlendirip "
        "bana uygun olup olmadığını söyle ve önerilerde bulun. Yanıtını sadece düz metin olarak ver."
    )
    run_assistant(default_query)