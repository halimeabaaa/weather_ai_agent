import os
import uuid
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.tools import Tool
from langchain import hub
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

# --- Kurulum ---
load_dotenv()
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
app = Flask(__name__)
sessions = {}

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# --- Firestore’dan kullanıcı profili ---
def get_user_profile_from_firestore(username: str) -> dict:
    try:
        users_ref = db.collection('kullanicilar')
        query = users_ref.where('isim', '==', username).limit(1)
        results = query.stream()
        doc = next(results, None)
        if not doc:
            return {"error": "Kullanıcı bulunamadı."}

        user_data = doc.to_dict()
        return {
            "name": user_data.get("isim"),
            "city": user_data.get("city"),
            "likedMinTemp": user_data.get("SevilenMinSicaklik"),
            "likedMaxTemp": user_data.get("SevilenMaxSicaklik"),
            "liked_conditions": user_data.get("sevilenHava", []),
            "disliked_conditions": user_data.get("sevilmeyenHava", []),
            "disease": user_data.get("sahipOlunanHastalik", [])
        }
    except Exception as e:
        return {"error": f"Firestore hatası: {e}"}

# --- Tools ---
google_search = DuckDuckGoSearchRun()
weather_search = OpenWeatherMapAPIWrapper()

tools = [
    Tool(
        name="Weather",
        func=weather_search.run,
        description="Belirtilen bir şehirdeki mevcut hava durumunu almak için kullanılır."
    ),
    Tool(
        name="GoogleSearch",
        func=google_search.run,
        description="Hava durumu ve kullanıcı profili dışındaki konular için kullanılır."
    ),
    Tool(
        name="GetUserProfile",
        func=get_user_profile_from_firestore,
        description="Kullanıcı profil bilgilerini almak için kullanılır."
    )
]

# --- Prompt ---
prompt_template = hub.pull("hwchase17/react")
system_instructions = (
    "Sen çok yardımsever bir asistansın. Görevin, kullanıcının seçtiği şehirdeki hava durumunu anlayıp yorumlamaktır.\n\n"

    "1. Adım: Kullanıcının kim olduğunu öğren.\n"
    "- Kullanıcı adını al ve 'GetUserProfile' aracını kullanarak Firestore’dan profil bilgilerini getir.\n"
    "- Profildeki bilgiler: hangi sıcaklıkları seviyor, hangi hava koşullarını seviyor veya sevmiyor, varsa alerji ve hastalık bilgileri.\n\n"

    "2. Adım: Hangi şehirde olduğunu öğren.\n"
    "- Artık şehir bilgisi Firestore’dan gelmeyecek.\n"
    "- Mobil uygulama kullanıcının harita üzerinden seçtiği şehri gönderir. Bu bilgiyi kullan.\n\n"

    "3. Adım: Şehirdeki hava durumunu kontrol et.\n"
    "- 'Weather' aracını kullan ve seçilen şehirdeki mevcut hava durumunu al.\n"
    "- Hava durumu verileri: sıcaklık, nem, rüzgar hızı ve hava koşulları (güneşli, yağmurlu, kar vb.).\n\n"

    "4. Adım: Hava durumu ile kullanıcı profilini karşılaştır.\n"
    "- Sıcaklık sevdiği aralıkta mı?\n"
    "- Hava koşulları onun hoşuna gidiyor mu?\n"
    "- Alerjisi veya hastalığı varsa havanın buna uygun olup olmadığını kontrol et.\n\n"

    "5. Adım: Kullanıcı için havanın uygunluğunu yüzde ile hesapla (%0-100).\n"
    "- Tamamen uygun ise %90-100\n"
    "- Kısmen uygun ise %50-70\n"
    "- Uygun değil ise %0-30\n\n"

    "6. Adım: Kullanıcıya önerilerde bulun.\n"
    "- Örnek: 'Bugün dışarı çıkmak için hava çok uygun', 'Şemsiye almayı unutma', 'Polen durumu yüksek, dikkat et' vb.\n"
    "- Alerjisi veya hastalığı varsa uyarıları buna göre ver.\n\n"

    "7. Adım: Yanıtını sadece düz metin olarak ver.\n"
    "- Markdown, sembol veya özel biçimlendirme kullanma.\n\n"

    "8. Adım: Sohbet geçmişini hatırla.\n"
    "- Eğer kullanıcıyla daha önce konuştuysan önceki mesajları göz önünde bulundur.\n"
    "- Cevaplarını tutarlı ve anlaşılır yap."
)

prompt_template.template = system_instructions + "\n\n" + prompt_template.template

# --- Agent Executor ---
def create_agent_executor(memory):
    agent = create_react_agent(llm, tools, prompt_template)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return executor

# --- API Routes ---
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('city'):
        return jsonify({"error": "'username' ve 'city' zorunludur."}), 400

    session_id = str(uuid.uuid4())
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        chat_memory=ChatMessageHistory(),
        return_messages=True
    )
    agent_executor = create_agent_executor(memory)
    sessions[session_id] = agent_executor

    initial_input = f"Kullanıcım '{data['username']}' için '{data['city']}' şehrinin hava durumunu analiz et."
    try:
        response = agent_executor.invoke({"input": initial_input})
        ai_response = response.get('output', 'Bir hata oluştu.')
        return jsonify({"response": ai_response, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": f"Ajan hatası: {e}"}), 500


@app.route('/continue_conversation', methods=['POST'])
def continue_conversation():
    data = request.get_json()
    if not data or not data.get('session_id') or not data.get('input'):
        return jsonify({"error": "'session_id' ve 'input' zorunludur."}), 400

    agent_executor = sessions.get(data['session_id'])
    if not agent_executor:
        return jsonify({"error": "Geçersiz oturum."}), 404

    try:
        response = agent_executor.invoke({"input": data['input']})
        ai_response = response.get('output', 'Bir hata oluştu.')
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": f"Ajan hatası: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
