import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 여기에 Streamlit 앱 코드를 작성하세요
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 페이지 기본 설정
st.set_page_config(page_title="FAQ 스마트 챗봇", page_icon="💬", layout="wide")

# 2. 실무 활용을 위한 FAQ 데이터 준비
faq_data = [
    {"q": "관리비 납부 마감일은 언제인가요?", "a": "매월 25일입니다. 휴일인 경우 다음 영업일로 연기됩니다."},
    {"q": "층간소음 민원은 어떻게 접수하나요?", "a": "관리사무소(02-123-4567)로 전화하시거나 앱 내 '민원 접수' 게시판을 이용해 주세요."},
    {"q": "주차 차량 등록은 어떻게 하나요?", "a": "차량등록증과 신분증을 지참하여 관리사무소에 방문하시면 즉시 등록 가능합니다."},
    {"q": "재활용 쓰레기 배출 요일은 언제인가요?", "a": "매주 화요일과 금요일 오전 6시부터 오후 11시까지입니다."},
    {"q": "엘리베이터 이사 사용 예약은 어떻게 하나요?", "a": "최소 3일 전까지 관리사무소에 예약하시고 엘리베이터 사용료를 납부하셔야 합니다."},
    {"q": "누수가 발생했는데 어떻게 해야 하나요?", "a": "즉시 관리사무소 시설팀으로 연락 주시면, 1차 현장 확인 후 조치 방법을 안내해 드립니다."}
]

faq_questions = [item['q'] for item in faq_data]
faq_answers = [item['a'] for item in faq_data]

# 3. 모델 및 FAISS 인덱스 로딩 (앱 구동 시 최초 1회만 실행되도록 캐싱)
@st.cache_resource
def load_chatbot_engine():
    # 한국어 처리에 뛰어난 다국어 SBERT 모델 사용
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    
    # FAQ 질문들을 벡터로 변환
    embeddings = model.encode(faq_questions).astype('float32')
    
    # FAISS 코사인 유사도 검색을 위한 L2 정규화 및 인덱스 생성
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return model, index

model, index = load_chatbot_engine()

# 4. 사이드바 UI (FAQ 목록 표시)
with st.sidebar:
    st.title("📋 등록된 FAQ 목록")
    st.markdown("현재 AI가 학습한 질문 리스트입니다.")
    for idx, q in enumerate(faq_questions):
        st.info(f"{idx+1}. {q}")

# 5. 메인 화면 UI (채팅창)
st.title("💬 스마트 AI 민원 상담 챗봇")
st.caption("질문을 입력하시면 AI가 가장 비슷한 FAQ 답변을 찾아 즉시 안내해 드립니다.")

# 대화 기록을 저장할 빈 공간(session_state) 생성
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 점이 궁금하신가요?"}]

# 이전 대화 기록들을 화면에 쭉 뿌려주기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. 사용자 입력 및 AI 답변 처리 로직
if user_query := st.chat_input("여기에 질문을 입력하세요... (예: 쓰레기 언제 버려요?)"):
    
    # 사용자의 질문을 채팅창에 띄우고 기록에 저장
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 답변 생성 과정 (스피너 애니메이션)
    with st.chat_message("assistant"):
        with st.spinner("AI가 가장 적합한 답변을 찾는 중입니다..."):
            
            # 사용자 질문을 벡터로 변환 후 검색
            query_vec = model.encode([user_query]).astype('float32')
            faiss.normalize_L2(query_vec)
            scores, indices = index.search(query_vec, 1)
            
            best_score = float(scores[0][0])
            best_idx = indices[0][0]
            
            # 임계값(0.4) 판별 및 답변 구성
            if best_score >= 0.4:
                answer = faq_answers[best_idx]
                response_text = f"{answer}\n\n*(📊 유사도 점수: **{best_score:.2f}**)*"
            else:
                response_text = f"죄송합니다. 해당 내용과 관련된 FAQ가 없습니다. 관리사무소로 직접 문의해 주시기 바랍니다.\n\n*(📊 가장 높은 유사도: **{best_score:.2f}** - 기준치 미달)*"
            
            # AI 답변을 채팅창에 띄우고 기록에 저장
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})


