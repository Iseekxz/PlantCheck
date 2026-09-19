import os
import csv
import nest_asyncio
import requests
from datetime import datetime
from PIL import Image
from google import genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from ultralytics import YOLO

nest_asyncio.apply()

# --- КЛЮЧИ ---
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ИЗ_BOTFATHER"
GEMINI_API_KEY = "ВАШ_КЛЮЧ_GEMINI"
WEATHER_API_KEY = "ВАШ_КЛЮЧ_ПОГОДЫ"
ADMIN_ID = 123456789

model = YOLO(MODEL_PATH)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- 1. ФУНКЦИЯ ПОГОДЫ ---
def get_weather():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Kokshetau&appid={WEATHER_API_KEY}&units=metric&lang=ru"
        res = requests.get(url).json()
        temp = res["main"]["temp"]
        desc = res["weather"][0]["description"]
        return f"{temp}°C, {desc}"
    except:
        return "Данные о погоде временно недоступны"

# --- 2. ФУНКЦИЯ ЛОГИРОВАНИЯ (CSV) ---
def log_to_csv(user_id, disease, conf):
    file_path = '/content/statistics.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Дата', 'ID Пользователя', 'Болезнь', 'Уверенность %'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, disease, round(conf, 1)])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
         InlineKeyboardButton("🇰🇿 Қазақша", callback_data="lang_kz")]
    ]
    await update.message.reply_text(
        "🌾 Добро пожаловать в Агро-Ассистент!\nВыберите язык / Тілді таңдаңыз:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data.startswith("lang_"):
        lang = query.data.split('_')[1]
        context.user_data['lang'] = lang
        msg = "✅ Язык установлен на Русский." if lang == "ru" else "✅ Тіл Қазақша болып орнатылды."
        await query.edit_message_text(text=f"{msg} Отправьте фото листа или задайте вопрос текстом.")

    # --- 3. РАСШИРЕННАЯ ОБРАБОТКА ВЫЗОВА ЭКСПЕРТА ---
    elif query.data == "call_expert":
        user = query.from_user
        last_ctx = context.user_data.get('last_context', None)
        username_str = f"@{user.username}" if user.username else "без юзернейма"
        
        admin_header = f"🚨 **Заявка на консультацию!**\nПользователь: {username_str} (ID: `{user.id}`)\n"

        if last_ctx and last_ctx.get('type') == 'photo':
            # Если запрос был по фотографии
            await context.bot.send_photo(
                chat_id=ADMIN_ID,
                photo=last_ctx['photo_id'],
                caption=f"📸 **Фото от фермера {username_str}**"
            )
            
            text_to_admin = (
                f"{admin_header}\n"
                f"🏷 **Диагноз YOLO:** `{last_ctx['diagnosis']}`\n\n"
                f"🤖 **Ответ ИИ:**\n{last_ctx['ai_response']}\n\n"
                f"👇 Ответьте фермеру командой:\n`/reply {user.id} Ваш текст`"
            )
            await context.bot.send_message(chat_id=ADMIN_ID, text=text_to_admin, parse_mode="Markdown")

        elif last_ctx and last_ctx.get('type') == 'text':
            # Если запрос был по текстовому вопросу
            text_to_admin = (
                f"{admin_header}\n"
                f"❓ **Вопрос фермера:**\n{last_ctx['user_text']}\n\n"
                f"🤖 **Ответ ИИ:**\n{last_ctx['ai_response']}\n\n"
                f"👇 Ответьте фермеру командой:\n`/reply {user.id} Ваш текст`"
            )
            await context.bot.send_message(chat_id=ADMIN_ID, text=text_to_admin, parse_mode="Markdown")

        else:
            # Базовое сообщение, если контекст не найден
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"{admin_header}\n👇 Ответьте командой:\n`/reply {user.id} Ваш текст`",
                parse_mode="Markdown"
            )

        await query.edit_message_text("📞 Ваша заявка и история вопроса переданы эксперту. Специалист свяжется с вами в течение часа.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang', 'ru')
    msg = await update.message.reply_text("🔎 Анализирую изображение и погодные условия...")

    file_path = f"/content/temp_crop_{update.message.message_id}.jpg"
    photo_obj = update.message.photo[-1]

    try:
        photo = await photo_obj.get_file()
        await photo.download_to_drive(file_path)

        # 1. Сначала YOLO делает свое предсказание
        results = model(file_path)
        top1_class = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf) * 100

        if confidence < 40.0: # Снизили порог, так как Gemini теперь фильтрует ошибки
            await msg.edit_text("⚠️ Изображение слишком размытое или нейросеть не может распознать объект. Сделайте более четкое фото.")
            return

        log_to_csv(update.message.from_user.id, top1_class, confidence)
        weather = get_weather()
        llm_lang = "русском языке" if lang == "ru" else "казахском языке (қазақша)"

        # 2. Открываем картинку для передачи в Gemini
        img = Image.open(file_path)

        # 3. Умный промпт: просим Gemini проверить диагноз YOLO
        prompt = f"""
        Ты главный агроном-эксперт. Узкоспециализированная модель предположила, что на фото болезнь: '{top1_class}' (уверенность {confidence:.1f}%).
        
        ВНИМАТЕЛЬНО ПОСМОТРИ НА ПРИКРЕПЛЕННОЕ ФОТО.
        
        СЦЕНАРИЙ А (Растение здоровое):
        Если на фото здоровое растение (нет явных признаков болезни) или вообще другой объект (человек, машина), ПРОИГНОРИРУЙ модель и ответь СТРОГО НА {llm_lang}:
        ✅ **Статус:** Растение выглядит здоровым (или опиши, что видишь).
        🌱 **Рекомендация:** [Дай краткий совет по базовому уходу, учитывая погоду: {weather}].
        
        СЦЕНАРИЙ Б (Растение действительно больное):
        Если признаки болезни есть, ответь СТРОГО НА {llm_lang} по шаблону:
        🚨 **Болезнь:** [Уточни название болезни]
        💊 **Препараты:** [Действующие вещества]
        🌦 **Сроки с учетом погоды:** [Когда опрыскивать, учитывая {weather}]
        
        Не пиши никаких вводных слов, сразу выдавай результат по нужному сценарию.
        """

        # Передаем и картинку (img), и текст (prompt)
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=[img, prompt]
        )
        
        keyboard = [[InlineKeyboardButton("👨‍🌾 Вызвать эксперта", callback_data="call_expert")]]

        # Сохраняем контекст фото для вызова эксперта
        context.user_data['last_context'] = {
            'type': 'photo',
            'photo_id': photo_obj.file_id,
            'diagnosis': f"YOLO: {top1_class} | ИИ-Проверка: {response.text[:30]}...",
            'ai_response': response.text
        }

        # Выводим ответ напрямую от Gemini (он сам выберет нужный шаблон)
        await msg.edit_text(
            response.text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            await msg.edit_text("⏳ Лимит запросов к ИИ временно превышен. Пожалуйста, подождите 1 минуту и повторите попытку.")
        else:
            await msg.edit_text(f"❌ Ошибка API: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка любых текстовых вопросов от пользователя"""
    user_text = update.message.text
    msg = await update.message.reply_text("⏳ Формирую ответ...")

    prompt = f"""
    Ты эрудированный агроном-консультант. Ответь на вопрос пользователя.
    Если вопрос касается сельского хозяйства, растений, погоды или препаратов - дай подробный профессиональный ответ.
    Если вопрос на отвлеченную тему, ответь вежливо и кратко.
    Вопрос пользователя: {user_text}
    """
    
    try:
        response = ai_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        keyboard = [[InlineKeyboardButton("👨‍🌾 Вызвать эксперта", callback_data="call_expert")]]

        # Сохраняем контекст текстового вопроса для вызова эксперта
        context.user_data['last_context'] = {
            'type': 'text',
            'user_text': user_text,
            'ai_response': response.text
        }

        await msg.edit_text(
            response.text, 
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception as e:
        await msg.edit_text("❌ Извините, сервис генерации ответов сейчас недоступен.")

async def reply_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для админа, чтобы отвечать пользователям через бота"""
    if update.message.from_user.id != ADMIN_ID:
        return 

    try:
        target_user_id = int(context.args[0])
        reply_text = " ".join(context.args[1:])

        await context.bot.send_message(
            chat_id=target_user_id,
            text=f"👨‍🌾 **Ответ от агронома:**\n{reply_text}",
            parse_mode="Markdown"
        )
        await update.message.reply_text("✅ Сообщение успешно отправлено фермеру.")
    except (IndexError, ValueError):
        await update.message.reply_text("❌ Ошибка формата. Используйте: /reply ID текст")

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button_handler))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.add_handler(CommandHandler("reply", reply_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

print("🚀 Бот запущен!")
app.run_polling()
