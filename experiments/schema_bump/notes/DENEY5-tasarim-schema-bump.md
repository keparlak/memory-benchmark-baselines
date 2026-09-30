# Deney 5 — Schema-bump update fidelity

Kaynak: @kartikb753, 22 Eyl 2026
> "i'd pair paraphrase with mid run tool schema bumps. if memory writes still target
> old arg shapes you're measuring lexical recall not whether updates survive drift."

## Soru

Bir agent koşusunun **ortasında** tool şeması değişince hafıza katmanı uyum sağlıyor mu?
Ve başarısızlık **write path'te mi read path'te mi**?

İkinci soru asıl katkı. Makalede şunu yazdım ve ölçmediğimi açıkça belirttim:

> "the write path is the problem, and the industry has been optimising the read path."

Bu deney o cümleyi hipotez olmaktan çıkarır ya da çürütür.

## Neden bu test, MemoryAgentBench'ten iyi

| | MAB FactConsolidation | Bu test |
|---|---|---|
| ground truth | seri numarası kurgusu | şemayı biz değiştiriyoruz |
| puanlama | string eşleşme, yorum payı var | JSON Schema validation, yorum yok |
| gerçeklik | tamamen sentetik | API versiyonlama üretimde sürekli oluyor |
| write/read ayrımı | yok | var |

Benim paraphrase eğrimde "eşdeğer ifade" kararı bana aitti — bu bir zayıflıktı.
Burada bir çağrı ya v2'ye valid ya değil.

## Kurulum

N görev × K adım. Tool seti T, başlangıçta şema **v1**.
Adım `k*`'de şema **v2**'ye geçer ve bu agent'a bir `tool_update` olayı olarak bildirilir
(gerçekte de böyle olur — changelog, 4xx hatası, deprecation uyarısı).

Şema değişim tipleri — her biri farklı bir hata imzası üretir:

| tip | örnek | ne yakalar |
|---|---|---|
| rename | `title` → `subject` | saf leksikal tutunma |
| split | `body` → `description` + `summary` | yapısal yeniden eşleme |
| add-required | yeni zorunlu `team` | eksik bilgi mi, eski şablon mu |
| type-change | `priority: str` → `severity: int` | tip körlüğü |
| remove | alan kaldırılır | ölü alanı taşımaya devam |

### Kollar

| kol | ne | maliyet |
|---|---|---|
| A | no-memory, tam bağlam (kontrol) | LLM |
| B | Mem0 (NIM'e yönlendirilmiş) | LLM |
| C | naive RAG — TF-IDF top-k | sıfır |
| D | deterministic policy: "en son şema kazanır" | sıfır |
| E | sahte-ajan (bağlamdaki en son şemayı kullanır) | sıfır, iskelet doğrulaması |

D kolu tavanı verir: policy doğruysa skor ~%100 olmalı. FactConsolidation'da
kapsam içi %96.6–100 çıkmıştı — aynı ayrımın burada da tutup tutmadığını görürüz.

## İkinci eksen — retry delta

Kaynak: @sukin_s (Creator of Nemp Memory), 22 Eyl 2026
> "I care about one number: same task, memory on vs off, after the agent has
> already failed once. If that delta is flat, the layer isn't memory."

Kartik'in ekseni *bilgi güncelleniyor mu* diye soruyor. Bu eksen *hafıza bir
sonraki denemeyi iyileştiriyor mu* diye soruyor — yani kategorinin kendi vaadini.
MemoryAgentBench bunu ölçmüyor; ilk geçiş performansını ölçüyor.

Korpus zaten bu şekle sahip: şema değişiminden sonra ajan eski şekille çağırıp
başarısız olur. Eksik olan tek şey retry döngüsü.

```
k* adiminda sema degisir
  -> ajan v1 sekliyle cagirir  -> FAIL (4xx + hata mesaji)
  -> hata olayi hafizaya yazilir
  -> ajan tekrar dener
retry_delta = success@retry(memory ON) - success@retry(memory OFF)
```

`retry_delta ≈ 0` ise katman hafıza değil, sadece depolama.

Bu ölçüm write/read attribution'ı tamamlar:
- attribution *nerede* bozulduğunu söyler
- retry_delta *önemi olup olmadığını* söyler

Bir katman attribution'da temiz görünüp retry_delta'da düz çıkabilir — o zaman
bilgiyi doğru saklıyor ama ajana bir faydası yok demektir. Bu da bir sonuçtur.

**Durum:** NVIDIA key geldiğinde schema-bump ile aynı koşuda ölçülecek.
Sahte-ajanla ölçmenin anlamı yok — deterministik ajan hatadan "öğrenmez",
sadece bağlamda ne varsa onu kullanır. Bu eksen gerçek model gerektirir.

## Ölçümler

1. **`call_validity@v2`** — `k*` sonrası çağrıların v2'ye uyum oranı. Ana skor.
2. **`stale_write`** — hafızanın sakladığı/konsolide ettiği kayıtlarda v1 alan
   adlarının hayatta kalma oranı. → **write path**
3. **`retrieval_recall@v2`** — hafızanın döndürdüğü bağlamda v2 tanımının bulunma
   oranı. → **read path**
4. **attribution tablosu** — (2)×(3) çapraz tablosu:

   | | v2 retrieved | v2 not retrieved |
   |---|---|---|
   | **write temiz** | başarısızlık ajanda | read path bozuk |
   | **write kirli** | write path bozuk | ikisi de |

(4) bu deneyin var oluş sebebi. 50 puanlık düşüşü iki yola ayıran şey bu tablo.

## Aşamalar

**Aşama 1 — LLM'siz (maliyet sıfır)**
- korpus üreteci (görev + şema v1/v2 çiftleri + trajectory)
- JSON Schema puanlayıcı
- E kolu (sahte-ajan) ile uçtan uca doğrulama
- C ve D kolları

Aşama 1 bitmeden LLM harcanmaz. İskelet sahte-ajanla çalışmıyorsa
gerçek modelle de çalışmaz.

**Aşama 2 — LLM'li**
- A kolu (no-memory kontrol)
- B kolu (Mem0, `openai_base_url` → `https://integrate.api.nvidia.com/v1`)

## Altyapı

- Endpoint: `https://integrate.api.nvidia.com` + `POST /v1/chat/completions`
  (OpenAI uyumlu) — https://docs.api.nvidia.com/nim/reference/llm-apis
- Mem0 yönlendirme: OpenAI provider'ı `openai_base_url` alıyor
  — https://docs.mem0.ai/components/llms/config
- Embedder: Mem0 varsayılanı OpenAI. Lokal bir embedder'a (sentence-transformers)
  çevrilecek ki key sadece LLM çağrıları için harcansın.

**Neden lokal opencode proxy'si değil:** `127.0.0.1:8045` ayakta değil, ayrı süreç
yönetimi gerektiriyor ve en önemlisi lokal proxy'e bağlı sonuç repro edilemez.
Şimdiye kadarki her ölçümün ayırt edici özelliği başkasının koşabilmesiydi.

## Kurulum tuzağı — Mem0'ı haksız yere suçlamak

İlk Mem0 koşusunda ürün **hiçbir şey saklamadı**: `stored_v2=False`, 0 kayıt.
Rakam olarak bu, `lossy-consolidate` karikatürümle aynı kutuya düşüyordu ve
"Mem0'ın write path'i tamamen bozuk" diye raporlanabilirdi.

Gerçek sebep bende: Mem0'ın fact-extraction adımı JSON bekliyor, ben ona
reasoning yapan bir model bağladım, model uzun prompt'ta düşünme bütçesini
tüketip **boş content** döndürdü. Log'daki iz:

```
Error parsing extraction response: Expecting value: line 1 column 1 (char 0)
```

`char 0` = boş string. Model JSON üretemiyor değil — üretebiliyor, ayrı ölçtüm;
üç farklı modda da düz JSON döndü. Ona yer bırakmayan bendim.

**Kural:** bir ürün bu harness'ta sıfır alırsa, önce kurulumun kendisinden
şüphelen. Ürünün başarısızlığı ile entegrasyonun başarısızlığı aynı sayıyı
üretir; ayıran tek şey log'a bakmaktır. Yayımlanacak her sıfır için bu kontrol
yapılacak.

Kurulum düzeldikten sonra da aynı sıfır bir kez daha çıktı — bu sefer sebep
**ölçütün kendisiydi**.

### Ölçüt katılığı: Mem0'ı ikinci kez haksız yere sıfırlamak

`stored_v2` başta şu regex'e bakıyordu:

```python
SIG = re.compile(r"Guncel imza -> (\w+)\(([^)]*)\)")
```

Yani benim korpusumun yazdığı öneki arıyordu. Ama Mem0 bir depo değil; gelen
metni fact-extraction ile **yeniden yazıyor**. Sakladığı kayıt şuydu:

> "User was informed that the 'priority' parameter was removed from
> create_ticket and replaced with 'severity' (int), updating the function
> signature to `create_ticket(title: str, body: str, severity: int)`"

İmza tam olarak orada. Önek yok. Ölçüt "hiç saklamamış" diyordu.

Düzeltme: ölçüt önek değil **içerik** arar — tool adı + içinde en az bir
`ad: tip` çifti olan parantez. Gevşetme yanlış pozitif üretmiyor, çünkü
`stored_v2` alan kümesinin v2 ile tam eşleşmesini istiyor; v1 imzası v2
sayılmaz. Doğrulandı: çağrı kayıtları eşleşmiyor, Aşama 1 sonuçları değişmedi.

**Genel ders:** bir ürünü kendi çıktı biçiminle ölçme. Ölçüt, ölçülen şeyin
biçimine değil **anlamına** bakmalı. Aksi halde farklı kelimelerle doğru işi
yapan bir sistem, sıfır alır — ve o sıfır tezini destekliyorsa fark etmezsin.

### Üçüncü tekrar: ortam değişkeni açık konfigürasyonu eziyor

Çoklu model karşılaştırması için `.env`'e bir OpenRouter anahtarı eklendi.
Sonraki Mem0 koşusunda **40 koşunun tamamı** `402 Insufficient credits` ile
düştü — oysa Mem0'a açıkça NVIDIA verilmişti.

Sebep `mem0/llms/openai.py` içinde:

```python
if os.environ.get("OPENROUTER_API_KEY"):      # Use OpenRouter
    self.client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                         base_url=...openrouter...)
else:
    api_key  = self.config.api_key or os.getenv("OPENAI_API_KEY")
    base_url = self.config.openai_base_url or ...
```

Ortamda o değişken varsa `from_config()` ile verilen `api_key` ve
`openai_base_url` **hiç okunmuyor**. Doğrulama:

```
verilen : openai_base_url = https://integrate.api.nvidia.com/v1
gerçek  : client.base_url = https://openrouter.ai/api/v1/
```

Düzeltme: `.env` okunuyor ama `OPENROUTER_*` ortama konmuyor
(`deney5_llm.SECRETS`). Sonra doğrulandı: `client.base_url` NVIDIA.

**Neden bu üçüncüsü en sinsi:** ilk ikisinde hata mesajı ürünü işaret
ediyordu ve ben yanlış okuyordum. Burada hata mesajı ürünü hiç işaret
etmiyordu; sessizce başka bir sağlayıcıya gidiliyordu. Bir kol sıfır alınca
"ürün başarısız" demeden önce **gerçekte hangi uca gittiğini** doğrulamak
gerekiyor — config'e ne yazdığını değil.

Not: bu, önceki Mem0 sonuçlarını geçersiz kılmaz. O koşularda `.env`'de
OpenRouter anahtarı yoktu, `else` dalı çalışıyordu. 261 ADD / 0 UPDATE
bulgusu NVIDIA üzerinden alındı.

## Dürüstlük notu

Bu deney bir hipotezi test etmek için tasarlandı ve hipotez benim.
Sonuç hipotezi çürütürse — yani write path temiz çıkıp başarısızlık read path'te
ya da ajanda toplanırsa — o sonuç da aynen yayımlanacak.
Aksi halde bu deney bir ölçüm değil, bir savunma olur.
