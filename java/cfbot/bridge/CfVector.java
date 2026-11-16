package cfbot.bridge;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.crforge.core.battle.data.BattleRecords;
import org.crforge.core.battle.data.GameRow;
import org.crforge.core.battle.data.GameTables;
import org.crforge.env.EnvConfig;
import org.crforge.env.LadderEnv;
import org.crforge.env.Schema;
import org.crforge.env.VectorEnv;
import org.crforge.env.Vocabulary;

/**
 * A batch of CRForge Ladder battles shown to the bot in its own observation layout.
 *
 * <p>Each battle is one {@link LadderEnv} bound to buffers of its own. A step gives both seats'
 * actions, then keeps stepping that battle with waits while neither seat has an open play (at most
 * {@code maxSkip} more decisions), as the bot's Rust bridge does on its simulator. Rewards, the
 * shaping terms, the public play history and the feature rows are all written here, on the worker
 * threads, into direct buffers that Python maps once.
 *
 * <p>Rows are {@code battle * 2 + seat}. Card tokens are the bot's: an index into the card row
 * names given at construction, 0 for none.
 */
public final class CfVector implements AutoCloseable {

  // ---- plane sizes: must match python/cfbot/crforge.py ----
  public static final int GLOBAL_F = 4;
  public static final int ENTITY_F = 24;
  public static final int ENTITY_X_F = 5;
  public static final int ENTITY_ROW = ENTITY_F + ENTITY_X_F;
  public static final int HAND_F = 4;
  public static final int DECK_ROWS = 8;
  public static final int DECK_F = 2;
  public static final int DECK_X_F = 3;
  public static final int PLAY_ROWS = 32;
  public static final int PLAY_F = 7;
  public static final int REVEAL_ROWS = 8;
  public static final int REVEAL_F = 3;
  public static final int REVEAL_X_F = 2;
  public static final int HAZARD_ROWS = 16;
  public static final int HAZARD_F = 5;
  public static final int EXTRA_F = 6;
  public static final int ABILITY_ROWS = 2;
  public static final int ABILITY_F = 9;
  public static final int PRIV_GLOBAL_F = 3;
  public static final int CARD_F = 28;
  public static final int UNIT_F = 16;
  /** The bot's flat action space; CRForge's 2307 actions are its first 2307. */
  public static final int ACTIONS = 4611;
  public static final int QUEUE_LEN = 4;

  // ---- one row of the float block ----
  public static final int O_GLOBAL = 0;
  public static final int O_ABILITY = O_GLOBAL + GLOBAL_F;
  public static final int O_HAND = O_ABILITY + ABILITY_ROWS * ABILITY_F;
  public static final int O_DECK = O_HAND + 4 * HAND_F;
  public static final int O_DECK_X = O_DECK + DECK_ROWS * DECK_F;
  public static final int O_PLAYS = O_DECK_X + DECK_ROWS * DECK_X_F;
  public static final int O_REVEAL = O_PLAYS + PLAY_ROWS * PLAY_F;
  public static final int O_REVEAL_X = O_REVEAL + REVEAL_ROWS * REVEAL_F;
  public static final int O_HAZARD = O_REVEAL_X + REVEAL_ROWS * REVEAL_X_F;
  public static final int O_EXTRAS = O_HAZARD + HAZARD_ROWS * HAZARD_F;
  public static final int O_PRIV_GLOBAL = O_EXTRAS + EXTRA_F;
  public static final int O_PRIV_HAND = O_PRIV_GLOBAL + PRIV_GLOBAL_F;
  public static final int O_PRIV_DECK = O_PRIV_HAND + 4;
  public static final int O_ENTITY = O_PRIV_DECK + DECK_ROWS;

  // ---- one row of the byte block ----
  public static final int B_PLAY_MASK = 0;
  public static final int B_REVEAL_MASK = B_PLAY_MASK + PLAY_ROWS;
  public static final int B_HAZARD_MASK = B_REVEAL_MASK + REVEAL_ROWS;
  public static final int B_LEGAL = B_HAZARD_MASK + HAZARD_ROWS;
  public static final int B_ENTITY_MASK = B_LEGAL + ACTIONS;

  private static final int SEAT = Schema.SEAT_INTS;
  private static final int CR_ACTIONS = Schema.ACTIONS;
  private static final int HEADER = Schema.HEADER_INTS;
  private static final int ROW = Schema.ENTITY_INTS;
  private static final float TICKS_PER_SECOND = 20f;
  /** Slot flags of a deck with no evolution and no hero slot. */
  private static final int[] PLAIN = new int[8];

  // Header offsets (Schema version 4).
  private static final int H_TICK = 0;
  private static final int H_ELIXIR = 2;
  private static final int H_HAND_REFILL_MS = 6;
  private static final int H_HAND_CARD = 7;
  private static final int H_HAND_COST = 11;
  private static final int H_HAND_PENDING = 15;
  private static final int H_DECK = 20;
  private static final int H_ENEMY_SEEN = 28;
  private static final int H_TOWERS = 36;
  private static final int H_ENTITY_COUNT = 48;
  private static final int H_ENTITY_OVERFLOW = 49;
  private static final int H_FULL_BAR_MS = 59;
  private static final int H_OVERTIME = 60;
  private static final int H_LAST_REFUSAL = 62;
  private static final int H_LAST_RUN_REFUSAL = 63;
  private static final int H_ABILITY = 101;
  private static final int H_ENEMY_PLAY_COUNT = 114;
  private static final int H_ENEMY_ELIXIR = 115;
  private static final int H_ENEMY_PLAYS = 116;
  private static final int H_DECK_LEVEL = 192;
  private static final int ENEMY_PLAY_ROWS = 8;
  private static final int ENEMY_PLAY_INTS = 9;

  // Entity row offsets.
  private static final int E_ID = 0;
  private static final int E_KIND = 1;
  private static final int E_SIDE = 2;
  private static final int E_ROW = 3;
  private static final int E_X = 4;
  private static final int E_Y = 5;
  private static final int E_HP = 6;
  private static final int E_MAX_HP = 7;
  private static final int E_SHIELD = 8;
  private static final int E_STATE = 9;
  private static final int E_FLAGS = 10;
  private static final int E_LEVEL = 11;
  private static final int E_DX = 12;
  private static final int E_DY = 13;
  private static final int E_DEPLOY_MS = 17;
  private static final int E_SPEED = 20;
  private static final int E_HIT_SPEED = 21;
  private static final int E_BUFF_COUNT = 22;
  private static final int E_BUFF = 23;
  private static final int E_RADIUS = 32;
  private static final int E_AIM_X = Schema.E_AIM_X;
  private static final int E_AIM_Y = Schema.E_AIM_Y;

  private static final int F_ALIVE = 1;
  private static final int F_HIDDEN = 2;
  private static final int F_INVISIBLE = 4;
  private static final int F_BUILDING = 32;
  private static final int F_KING = 64;
  private static final int F_CLONE = 128;
  private static final int F_CHARGED = 2048;

  private static final int BUFF_RAGE = 1;
  private static final int BUFF_SLOW = 2;
  private static final int BUFF_CURSE = 4;
  private static final int BUFF_GOBLIN_CURSE = 8;
  private static final int BUFF_VINES = 16;
  private static final int BUFF_STUN = 32;

  private final GameTables tables;
  private final Vocabulary vocabulary;
  private final EnvConfig config;
  private final int count;
  private final int rows;
  private final int threads;
  private final int decisionTicks;
  private final int maxSkip;
  private final int maxEntities;
  private final int level;
  private final float maxElixir;
  private final int floatRow;
  private final int byteRow;

  private final String[] cardRows;
  private final int[] cardOfVocab;
  private final int[] unitOfVocab;
  private final int[] hazardOfVocab;
  // Which hazards keep their row when more are in play than the plane holds, by vocabulary id:
  // 0 an area effect, 1 a projectile a card casts (a Fireball, a rolling Log), 2 a unit's shot,
  // 3 a projectile that is decoration only. A crown tower's shot is moved to 3 as it is read.
  private final int[] hazardTier;
  // The vocabulary id of the projectile a unit row shoots, -1 for none.
  private final int[] shotOfUnit;
  // For a card that casts a volley (Arrows: three waves of ten), how far apart in game units two
  // of its projectiles may aim and still be one cast; -1 for a card that casts one projectile.
  // The arrows aim well past the spell's circle, about two radii to each side of its middle.
  private final int[] castSpread;
  private final int[] towerOfVocab;
  private final int[] buffFlags;
  private final int nUnits;
  private final int nHazards;
  private final float[] cardTable;
  private final float[] unitTable;
  private final int[] cardCost;
  private final int[] cardKind;
  private final boolean[] cardChampion;
  /** Per card token: the vocabulary ids of the bodies it deploys and what one is worth. */
  private final int[][] cardBodies;
  private final float[] cardBodyValue;
  private final int[] defaultDeck;

  private final ByteBuffer floatBytes;
  private final ByteBuffer byteBytes;
  private final FloatBuffer floats;
  private final ByteBuffer bytes;
  private final ByteBuffer actionBytes;
  private final IntBuffer actions;
  private final ByteBuffer rowIntBytes;
  private final IntBuffer rowInts;
  private final ByteBuffer rowFloatBytes;
  private final FloatBuffer rowFloats;
  private final ByteBuffer rowFlagBytes;
  private final ByteBuffer envIntBytes;
  private final IntBuffer envInts;
  private final ByteBuffer envFlagBytes;
  private final ByteBuffer deckBytes;
  private final ShortBuffer decks;

  // Row ints: entity count, entity overflow, hazard overflow. Row floats: reward, trade, leak,
  // ticks, six tower fractions. Row flags: choice, done, truncated, verdict.
  public static final int ROW_INTS = 3;
  public static final int ROW_FLOATS = 10;
  public static final int ROW_FLAGS = 4;
  // Env ints: outcome, episode decisions, episode ticks. Env flags: queued, need next.
  public static final int ENV_INTS = 3;
  public static final int ENV_FLAGS = 2;

  private final Slot[] slots;
  private final ExecutorService pool;
  private final AtomicInteger cursor = new AtomicInteger();
  private final List<int[]> crashes = new ArrayList<>();
  private final AtomicInteger deckRetries = new AtomicInteger();
  private final List<int[]> refusals = new ArrayList<>();
  private final AtomicInteger runRefusals = new AtomicInteger();
  private final List<int[]> runRefused = new ArrayList<>();
  private volatile String lastError = "";

  /**
   * @param tablesFolder the game tables of one data version
   * @param cardRows the tables' card row name of each card token; index 0 and unused tokens null
   * @param defaultDeck eight card tokens every battle can fall back to
   */
  public CfVector(
      String tablesFolder,
      int count,
      int threads,
      int decisionTicks,
      int maxSkip,
      int maxEntities,
      int level,
      int maxTicks,
      String[] cardRows,
      int[] defaultDeck) {
    if (count < 1 || decisionTicks < 1 || maxEntities < 16 || maxEntities > Schema.MAX_ENTITIES) {
      throw new IllegalArgumentException("bad vector size");
    }
    this.tables = VectorEnv.tables(Path.of(tablesFolder));
    this.vocabulary = VectorEnv.vocabulary(tables);
    this.config = new EnvConfig(decisionTicks, level, level, Math.max(0, maxTicks));
    this.count = count;
    this.rows = count * 2;
    this.threads = Math.max(1, Math.min(threads, count));
    this.decisionTicks = decisionTicks;
    this.maxSkip = maxSkip;
    this.maxEntities = maxEntities;
    this.level = level;
    this.maxElixir = vocabulary.maxElixir();
    this.floatRow = O_ENTITY + maxEntities * ENTITY_ROW;
    this.byteRow = B_ENTITY_MASK + maxEntities;
    this.cardRows = cardRows.clone();
    this.defaultDeck = defaultDeck.clone();

    int size = vocabulary.size();
    cardOfVocab = new int[size];
    for (int token = 1; token < cardRows.length; token++) {
      if (cardRows[token] != null && vocabulary.isDeckCard(cardRows[token])) {
        cardOfVocab[vocabulary.id(cardRows[token])] = token;
      }
    }
    unitOfVocab = dense(size, "characters", "buildings");
    hazardOfVocab = dense(size, "area_effect_objects", "projectiles");
    nUnits = max(unitOfVocab) + 1;
    nHazards = max(hazardOfVocab) + 1;
    hazardTier = new int[size];
    shotOfUnit = new int[size];
    castSpread = new int[size];
    Arrays.fill(castSpread, -1);
    hazardTiers(cardRows);
    towerOfVocab = new int[size];
    List<String> names = vocabulary.names();
    for (int id = 0; id < size; id++) {
      String name = names.get(id);
      towerOfVocab[id] =
          name.contains("Cannoneer") || name.contains("CannonTower")
              ? 2
              : name.contains("Knife") ? 3 : name.contains("Chef") ? 4 : 1;
    }
    List<String> buffs = vocabulary.buffNames();
    buffFlags = new int[buffs.size()];
    for (int i = 0; i < buffFlags.length; i++) {
      String name = buffs.get(i).toLowerCase(Locale.ROOT);
      int flags = 0;
      if (name.contains("goblincurse")) {
        flags |= BUFF_GOBLIN_CURSE;
      } else if (name.contains("curse")) {
        flags |= BUFF_CURSE;
      }
      if (name.contains("rage")) {
        flags |= BUFF_RAGE;
      }
      if (name.contains("slow") || name.contains("cold")) {
        flags |= BUFF_SLOW;
      }
      if (name.contains("vines") || name.contains("snare")) {
        flags |= BUFF_VINES;
      }
      if (name.contains("stun") || name.contains("freeze")) {
        flags |= BUFF_STUN;
      }
      buffFlags[i] = flags;
    }

    cardTable = new float[cardRows.length * CARD_F];
    unitTable = new float[nUnits * UNIT_F];
    cardCost = new int[cardRows.length];
    cardKind = new int[cardRows.length];
    cardChampion = new boolean[cardRows.length];
    cardBodies = new int[cardRows.length][];
    cardBodyValue = new float[cardRows.length];
    Facts.fill(this);

    floatBytes = direct(rows * floatRow * 4);
    floats = floatBytes.asFloatBuffer();
    byteBytes = direct(rows * byteRow);
    bytes = byteBytes;
    actionBytes = direct(rows * 4);
    actions = actionBytes.asIntBuffer();
    rowIntBytes = direct(rows * ROW_INTS * 4);
    rowInts = rowIntBytes.asIntBuffer();
    rowFloatBytes = direct(rows * ROW_FLOATS * 4);
    rowFloats = rowFloatBytes.asFloatBuffer();
    rowFlagBytes = direct(rows * ROW_FLAGS);
    envIntBytes = direct(count * ENV_INTS * 4);
    envInts = envIntBytes.asIntBuffer();
    envFlagBytes = direct(count * ENV_FLAGS);
    deckBytes = direct(count * 16 * 2);
    decks = deckBytes.asShortBuffer();

    slots = new Slot[count];
    for (int i = 0; i < count; i++) {
      slots[i] = new Slot(i);
    }
    pool =
        this.threads > 1
            ? Executors.newFixedThreadPool(
                this.threads - 1,
                task -> {
                  Thread thread = new Thread(task, "cfbot-env");
                  thread.setDaemon(true);
                  return thread;
                })
            : null;
  }

  private static ByteBuffer direct(int size) {
    return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
  }

  private static int max(int[] values) {
    int best = 0;
    for (int value : values) {
      best = Math.max(best, value);
    }
    return best;
  }

  /** Tokens 1.. for the sorted row names of these tables, by vocabulary id; 0 elsewhere. */
  private int[] dense(int size, String... tableNames) {
    TreeSet<String> names = new TreeSet<>();
    for (String table : tableNames) {
      for (GameRow row : tables.table(table).rows()) {
        names.add(row.name());
      }
    }
    int[] out = new int[size];
    int token = 1;
    for (String name : names) {
      out[vocabulary.id(name)] = token++;
    }
    return out;
  }

  /**
   * Ranks every hazard row. The plane holds 16 hazards and a battle can show more than that in
   * shots alone (a Hunter fires ten pellets, each crown tower one arrow after another), so the
   * rows go to what a player has to answer first: spell areas, then the projectiles cards cast,
   * then units' shots, and last the towers' own shots and arrows drawn for show.
   */
  private void hazardTiers(String[] cardRows) {
    Arrays.fill(shotOfUnit, -1);
    for (GameRow row : tables.table("projectiles").rows()) {
      hazardTier[vocabulary.id(row.name())] = row.name().endsWith("Deco") ? 3 : 2;
    }
    BattleRecords records = new BattleRecords(tables);
    for (String name : cardRows) {
      if (name == null || !vocabulary.isDeckCard(name)) {
        continue;
      }
      String cast;
      boolean volley;
      try {
        var card = records.card(name);
        cast = card.projectile();
        volley = card.multipleProjectiles() > 1 || card.projectileWaves() > 1;
      } catch (RuntimeException refused) {
        continue;
      }
      // A cast projectile and what it turns into: the Log thrown, then the Log rolling.
      for (int hops = 0; cast != null && hops < 4; hops++) {
        var data = records.projectile(cast);
        hazardTier[vocabulary.id(cast)] = 1;
        castSpread[vocabulary.id(cast)] = volley ? 4 * Math.max(data.radius(), 0) + 2000 : -1;
        cast = data.spawnProjectile();
      }
    }
    for (String table : new String[] {"characters", "buildings"}) {
      for (GameRow row : tables.table(table).rows()) {
        try {
          var shot = records.unit(row.name()).projectile();
          if (shot != null) {
            shotOfUnit[vocabulary.id(row.name())] = vocabulary.id(shot.name());
          }
        } catch (RuntimeException refused) {
          // A row the battle core does not build shoots nothing here.
        }
      }
    }
  }

  // ---- what Python reads ----

  public int count() {
    return count;
  }

  public int floatRow() {
    return floatRow;
  }

  public int byteRow() {
    return byteRow;
  }

  public int nUnits() {
    return nUnits;
  }

  public int nHazards() {
    return nHazards;
  }

  public int maxElixir() {
    return (int) maxElixir;
  }

  public String dataVersion() {
    return vocabulary.dataVersion();
  }

  public String vocabularyHash() {
    return vocabulary.hash();
  }

  public float[] cardTable() {
    return cardTable;
  }

  public float[] unitTable() {
    return unitTable;
  }

  public int[] cardCosts() {
    return cardCost;
  }

  /** Per card token: 0 none, 1 troop, 2 building, 3 spell, 4 other (Mirror, a variant card). */
  public int[] cardKinds() {
    return cardKind;
  }

  public boolean[] cardChampions() {
    return cardChampion;
  }

  /** Unit names by unit token, empty for token 0. */
  public String[] unitNames() {
    String[] out = new String[nUnits];
    Arrays.fill(out, "");
    List<String> names = vocabulary.names();
    for (int id = 0; id < unitOfVocab.length; id++) {
      if (unitOfVocab[id] > 0) {
        out[unitOfVocab[id]] = names.get(id);
      }
    }
    return out;
  }

  /** Hazard ranks by hazard token (see {@link #hazardTiers}); a crown tower's shot reads 2 here. */
  public int[] hazardRanks() {
    int[] out = new int[nHazards];
    for (int id = 0; id < hazardOfVocab.length; id++) {
      if (hazardOfVocab[id] > 0) {
        out[hazardOfVocab[id]] = hazardTier[id];
      }
    }
    return out;
  }

  /** Units' and towers' shots left out of the hazard plane for want of a row, over all battles. */
  public long shotsDropped() {
    long total = 0;
    for (Slot slot : slots) {
      total += slot.shotsDropped;
    }
    return total;
  }

  /** Hazard names by hazard token, empty for token 0. */
  public String[] hazardNames() {
    String[] out = new String[nHazards];
    Arrays.fill(out, "");
    List<String> names = vocabulary.names();
    for (int id = 0; id < hazardOfVocab.length; id++) {
      if (hazardOfVocab[id] > 0) {
        out[hazardOfVocab[id]] = names.get(id);
      }
    }
    return out;
  }

  public ByteBuffer floats() {
    return floatBytes;
  }

  public ByteBuffer bytes() {
    return byteBytes;
  }

  public ByteBuffer actions() {
    return actionBytes;
  }

  public ByteBuffer rowInts() {
    return rowIntBytes;
  }

  public ByteBuffer rowFloats() {
    return rowFloatBytes;
  }

  public ByteBuffer rowFlags() {
    return rowFlagBytes;
  }

  public ByteBuffer envInts() {
    return envIntBytes;
  }

  public ByteBuffer envFlags() {
    return envFlagBytes;
  }

  public ByteBuffer decks() {
    return deckBytes;
  }

  public int deckRetries() {
    return deckRetries.get();
  }

  public String lastError() {
    return lastError;
  }

  /** The decks of crashed battles, sixteen card tokens each, flattened. */
  public synchronized int[] crashes() {
    int[] out = new int[crashes.size() * 16];
    for (int i = 0; i < crashes.size(); i++) {
      System.arraycopy(crashes.get(i), 0, out, i * 16, 16);
    }
    return out;
  }

  /**
   * Plays the simulator refused although the seat's mask left them open: refusal code, card token
   * (0 for an ability), action, tick and the seat's elixir in ten-thousandths when it was given.
   * Five ints each, flattened; the first 4096 are kept.
   */
  public synchronized int[] refusals() {
    int[] out = new int[refusals.size() * 5];
    for (int i = 0; i < refusals.size(); i++) {
      System.arraycopy(refusals.get(i), 0, out, i * 5, 5);
    }
    return out;
  }

  /** Plays accepted when given that the match or the placement refused when they ran. */
  public int runRefusals() {
    return runRefusals.get();
  }

  /**
   * The first 4096 of them: the run's code (a match gate's, or 1000 for no placement), the token
   * of the seat's last card given, its action, and the tick. Four ints each, flattened.
   */
  public synchronized int[] runRefused() {
    int[] out = new int[runRefused.size() * 4];
    for (int i = 0; i < runRefused.size(); i++) {
      System.arraycopy(runRefused.get(i), 0, out, i * 4, 4);
    }
    return out;
  }

  // ---- matches ----

  /** Queues a match for one battle; it starts at that battle's next episode. */
  public void queueMatch(int index, int[] deckTokens, int seed) {
    checkDecks(deckTokens);
    slots[index].queue.add(new Match(deckTokens.clone(), seed, true));
  }

  /** Gives sampled matches for battles to start when they have nothing queued. */
  public void setNext(int[] indices, int[] deckTokens, int[] seeds) {
    for (int k = 0; k < indices.length; k++) {
      int[] deck = Arrays.copyOfRange(deckTokens, k * 16, k * 16 + 16);
      checkDecks(deck);
      slots[indices[k]].next = new Match(deck, seeds[k], false);
      envFlagBytes.put(indices[k] * ENV_FLAGS + 1, (byte) 0);
    }
  }

  private void checkDecks(int[] deck) {
    if (deck.length != 16) {
      throw new IllegalArgumentException("a match is two decks of eight card tokens");
    }
    for (int token : deck) {
      if (token <= 0 || token >= cardRows.length || cardRows[token] == null) {
        throw new IllegalArgumentException("no such card token: " + token);
      }
    }
  }

  /** Starts a new episode in every battle, dropping the running ones unreported. */
  public void reset() {
    run(slot -> slot.restart());
  }

  /** One decision in every battle, with forced waits skipped. Reads the action buffer. */
  public void step() {
    run(slot -> slot.step(actions.get(slot.index * 2), actions.get(slot.index * 2 + 1)));
  }

  private interface Task {
    void run(Slot slot);
  }

  private void run(Task task) {
    if (pool == null) {
      for (Slot slot : slots) {
        task.run(slot);
      }
      return;
    }
    cursor.set(0);
    CountDownLatch latch = new CountDownLatch(threads - 1);
    Throwable[] failure = new Throwable[1];
    Runnable work =
        () -> {
          try {
            for (int i = cursor.getAndIncrement(); i < count; i = cursor.getAndIncrement()) {
              task.run(slots[i]);
            }
          } catch (Throwable thrown) {
            failure[0] = thrown;
          }
        };
    for (int t = 0; t < threads - 1; t++) {
      pool.execute(
          () -> {
            try {
              work.run();
            } finally {
              latch.countDown();
            }
          });
    }
    work.run();
    try {
      latch.await();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(interrupted);
    }
    if (failure[0] != null) {
      throw new IllegalStateException("a battle could not be stepped", failure[0]);
    }
  }

  @Override
  public void close() {
    if (pool != null) {
      pool.shutdownNow();
    }
  }

  private record Match(int[] decks, int seed, boolean queued) {}

  // ---------------------------------------------------------------------------------------------

  /** One battle and what the bot tracks about it. */
  private final class Slot {
    final int index;
    final int[] obs = new int[2 * SEAT];
    final byte[] mask = new byte[2 * CR_ACTIONS];
    final float[] reward = new float[2];
    final int[] info = new int[Schema.INFO_INTS];
    final ArrayDeque<Match> queue = new ArrayDeque<>();
    volatile Match next;
    LadderEnv env;
    boolean queued;
    int decisions;
    int fallbackSeed;
    final int[] deckTokens = new int[16];

    // Per seat: its own cycle, and the plays it has seen.
    final int[] plays = new int[2];
    final int[][] lastPlay = new int[2][DECK_ROWS];
    /** Each seat's hand as its last written observation showed it, by card token. */
    final int[][] handShown = new int[2][4];
    final int[][] lastGiven = new int[2][2];
    final float[][] journal = new float[2][PLAY_ROWS * PLAY_F];
    final int[][] journalSeq = new int[2][PLAY_ROWS];
    final int[] journalSize = new int[2];
    final int[] enemySeen = new int[2];
    final int[][] seenLastSeq = new int[2][];
    final int[][] seenCount = new int[2][];
    final int[][] seenTick = new int[2][];
    final boolean[][] attributed = new boolean[2][64];

    // Per seat: the health of its own troops, by the seat's entity id, for trade accounting.
    final float[][] health = new float[2][256];
    final float[][] rate = new float[2][256];
    final int[][] stamp = new int[2][256];
    final int[][] active = new int[2][Schema.MAX_ENTITIES];
    final int[] activeCount = new int[2];
    int stampNow;
    final int[][] bodyRow = new int[2][32];
    final float[][] bodyValue = new float[2][32];
    final int[] bodyCount = new int[2];

    final float[] trade = new float[2];
    final float[] leak = new float[2];
    final float[][] scratch = new float[Schema.MAX_ENTITIES][ENTITY_ROW];
    final int[] order = new int[Schema.MAX_ENTITIES];
    final int[] lastEntityCount = new int[2];
    final int[] hazardAt = new int[Schema.MAX_ENTITIES];
    final int[] hazardRank = new int[Schema.MAX_ENTITIES];
    final int[] towerShot = new int[8];
    final int[] hazardVolley = new int[Schema.MAX_ENTITIES];
    final long[] volleyX = new long[Schema.MAX_ENTITIES];
    final long[] volleyY = new long[Schema.MAX_ENTITIES];
    final int[] volleySize = new int[Schema.MAX_ENTITIES];
    long shotsDropped;

    Slot(int index) {
      this.index = index;
      for (int seat = 0; seat < 2; seat++) {
        seenLastSeq[seat] = new int[cardRows.length];
        seenCount[seat] = new int[cardRows.length];
        seenTick[seat] = new int[cardRows.length];
      }
    }

    private void bind() {
      env = new LadderEnv(tables, vocabulary, config);
      env.bind(IntBuffer.wrap(obs), ByteBuffer.wrap(mask), FloatBuffer.wrap(reward), IntBuffer.wrap(info));
    }

    private List<String> names(int[] tokens, int from) {
      String[] out = new String[8];
      for (int i = 0; i < 8; i++) {
        out[i] = cardRows[tokens[from + i]];
      }
      return List.of(out);
    }

    private boolean begin(int[] tokens, int seed) {
      try {
        if (env == null || env.fault() != null) {
          bind();
        }
        List<String> deck0 = names(tokens, 0);
        List<String> deck1 = names(tokens, 8);
        env.checkDeck(deck0, PLAIN);
        env.checkDeck(deck1, PLAIN);
        env.reset(seed, deck0, deck1, PLAIN, PLAIN);
        if (env.fault() != null || env.done()) {
          lastError = String.valueOf(env.fault());
          env = null;
          return false;
        }
      } catch (RuntimeException | AssertionError | StackOverflowError thrown) {
        lastError = String.valueOf(thrown);
        env = null;
        return false;
      }
      System.arraycopy(tokens, 0, deckTokens, 0, 16);
      return true;
    }

    /** Starts the next episode: a queued match, else the sampled one, else the default decks. */
    void startEpisode() {
      Match match = queue.poll();
      if (match == null) {
        match = next;
        next = null;
        envFlagBytes.put(index * ENV_FLAGS + 1, (byte) 1);
      }
      fallbackSeed = fallbackSeed * 1664525 + 1013904223 + index;
      boolean started = match != null && begin(match.decks, match.seed);
      queued = started && match.queued;
      if (!started) {
        if (match != null) {
          deckRetries.incrementAndGet();
        }
        int[] both = new int[16];
        System.arraycopy(defaultDeck, 0, both, 0, 8);
        System.arraycopy(defaultDeck, 0, both, 8, 8);
        int seed = match != null ? match.seed : fallbackSeed;
        if (!begin(both, seed)) {
          throw new IllegalStateException("the default decks could not start a battle: " + lastError);
        }
      }
      decisions = 0;
      stampNow = 0;
      for (int seat = 0; seat < 2; seat++) {
        plays[seat] = 0;
        Arrays.fill(lastPlay[seat], -1);
        journalSize[seat] = 0;
        enemySeen[seat] = 0;
        Arrays.fill(seenLastSeq[seat], -1);
        Arrays.fill(seenCount[seat], 0);
        Arrays.fill(seenTick[seat], 0);
        Arrays.fill(attributed[seat], false);
        activeCount[seat] = 0;
        Arrays.fill(stamp[seat], -1);
        bodyCount[seat] = 0;
        for (int i = 0; i < 8; i++) {
          int token = deckTokens[seat * 8 + i];
          int[] bodies = cardBodies[token];
          if (bodies != null && cardBodyValue[token] > 0) {
            for (int body : bodies) {
              bodyRow[seat][bodyCount[seat]] = body;
              bodyValue[seat][bodyCount[seat]++] = cardBodyValue[token];
            }
          }
        }
      }
    }

    void restart() {
      startEpisode();
      clearStep();
      envInts.put(index * ENV_INTS, -1);
      envInts.put(index * ENV_INTS + 1, 0);
      envInts.put(index * ENV_INTS + 2, 0);
      envFlagBytes.put(index * ENV_FLAGS, (byte) 0);
      featurize(0f);
    }

    private void clearStep() {
      for (int seat = 0; seat < 2; seat++) {
        int row = index * 2 + seat;
        for (int k = 0; k < ROW_FLOATS; k++) {
          rowFloats.put(row * ROW_FLOATS + k, 0f);
        }
        rowFlagBytes.put(row * ROW_FLAGS + 1, (byte) 0);
        rowFlagBytes.put(row * ROW_FLAGS + 2, (byte) 0);
        rowFlagBytes.put(row * ROW_FLAGS + 3, (byte) 0);
      }
    }

    private int header(int seat, int field) {
      return obs[seat * SEAT + field];
    }

    private boolean choice(int seat) {
      int base = seat * CR_ACTIONS;
      for (int action = 1; action < CR_ACTIONS; action++) {
        if (mask[base + action] != 0) {
          return true;
        }
      }
      return false;
    }

    void step(int action0, int action1) {
      trade[0] = trade[1] = leak[0] = leak[1] = 0f;
      int[] given = {action0, action1};
      int[] verdict = new int[2];
      for (int seat = 0; seat < 2; seat++) {
        if (given[seat] < 0 || given[seat] >= CR_ACTIONS) {
          verdict[seat] = 3;
          given[seat] = 0;
        }
      }
      int startTick = header(0, H_TICK);
      int[] elixirBefore = {header(0, H_ELIXIR), header(1, H_ELIXIR)};
      boolean crashed = false;
      try {
        advance(given[0], given[1]);
        for (int seat = 0; seat < 2; seat++) {
          if (given[seat] != 0) {
            boolean accepted = header(seat, H_LAST_REFUSAL) == 0;
            verdict[seat] = accepted ? 1 : 2;
            if (accepted && given[seat] < Schema.FIRST_ABILITY) {
              ownPlay(seat, given[seat], startTick);
            }
            if (!accepted) {
              refused(seat, given[seat], startTick, elixirBefore[seat]);
            }
          }
        }
        decisions++;
        int skipped = 0;
        while (!env.done() && skipped < maxSkip && !choice(0) && !choice(1)) {
          advance(0, 0);
          skipped++;
        }
      } catch (RuntimeException | AssertionError | StackOverflowError thrown) {
        lastError = String.valueOf(thrown);
        crashed = true;
        env = null;
      }
      if (crashed || env == null || env.fault() != null) {
        crash();
        return;
      }

      boolean done = env.done();
      int winner = info[2];
      boolean truncated = done && winner < 0;
      int endTick = header(0, H_TICK);
      for (int seat = 0; seat < 2; seat++) {
        int row = index * 2 + seat;
        int at = row * ROW_FLOATS;
        rowFloats.put(at, done && winner >= 0 && winner < 2 ? (winner == seat ? 1f : -1f) : 0f);
        rowFloats.put(at + 1, trade[seat]);
        rowFloats.put(at + 2, leak[seat]);
        rowFloats.put(at + 3, endTick - startTick);
        for (int tower = 0; tower < 6; tower++) {
          int hp = header(seat, H_TOWERS + tower * 2);
          int full = header(seat, H_TOWERS + tower * 2 + 1);
          rowFloats.put(at + 4 + tower, full > 0 ? Math.min(1f, Math.max(0f, hp / (float) full)) : 0f);
        }
        rowFlagBytes.put(row * ROW_FLAGS + 1, (byte) (done ? 1 : 0));
        rowFlagBytes.put(row * ROW_FLAGS + 2, (byte) (truncated ? 1 : 0));
        rowFlagBytes.put(row * ROW_FLAGS + 3, (byte) verdict[seat]);
      }
      envInts.put(index * ENV_INTS, done && winner >= 0 ? winner : -1);
      envInts.put(index * ENV_INTS + 1, done ? decisions : 0);
      envInts.put(index * ENV_INTS + 2, done ? endTick : 0);
      if (done) {
        finish();
        startEpisode();
        featurize(0f);
      } else {
        featurize((endTick - startTick) / TICKS_PER_SECOND);
      }
    }

    private void finish() {
      for (int i = 0; i < 16; i++) {
        decks.put(index * 16 + i, (short) deckTokens[i]);
      }
      envFlagBytes.put(index * ENV_FLAGS, (byte) (queued ? 1 : 0));
    }

    private void refused(int seat, int action, int tick, int elixir) {
      int card = action < Schema.FIRST_ABILITY ? handShown[seat][(action - 1) / Schema.TILES] : 0;
      synchronized (CfVector.this) {
        if (refusals.size() < 4096) {
          refusals.add(new int[] {header(seat, H_LAST_REFUSAL), card, action, tick, elixir});
        }
      }
    }

    private void crash() {
      synchronized (CfVector.this) {
        if (crashes.size() < 4096) {
          crashes.add(deckTokens.clone());
        }
      }
      finish();
      clearStep();
      for (int seat = 0; seat < 2; seat++) {
        int row = index * 2 + seat;
        rowFloats.put(row * ROW_FLOATS + 3, decisionTicks);
        rowFlagBytes.put(row * ROW_FLAGS + 1, (byte) 1);
        rowFlagBytes.put(row * ROW_FLAGS + 2, (byte) 1);
      }
      envInts.put(index * ENV_INTS, 3);
      envInts.put(index * ENV_INTS + 1, 0);
      envInts.put(index * ENV_INTS + 2, 0);
      startEpisode();
      featurize(0f);
    }

    /** One decision of the battle, then the bookkeeping that has to see every decision. */
    private void advance(int action0, int action1) {
      int before = header(0, H_TICK);
      env.step(action0, action1);
      if (env.fault() != null) {
        return;
      }
      int ticks = header(0, H_TICK) - before;
      stampNow++;
      for (int seat = 0; seat < 2; seat++) {
        if (header(seat, H_ELIXIR) >= maxElixir * 10000f) {
          int bar = header(seat, H_FULL_BAR_MS);
          if (bar > 0) {
            leak[seat] += ticks * maxElixir * 50f / bar;
          }
        }
        if (header(seat, H_LAST_RUN_REFUSAL) != 0) {
          runRefusals.incrementAndGet();
          synchronized (CfVector.this) {
            if (runRefused.size() < 4096) {
              runRefused.add(new int[] {
                header(seat, H_LAST_RUN_REFUSAL), lastGiven[seat][0], lastGiven[seat][1], header(seat, H_TICK)
              });
            }
          }
        }
        damage(seat);
        enemyPlays(seat);
      }
    }

    /** Credits the other seat for health this seat's card-deployed troops lost. */
    private void damage(int seat) {
      int base = seat * SEAT;
      int n = Math.min(obs[base + H_ENTITY_COUNT], Schema.MAX_ENTITIES);
      for (int e = 0; e < n; e++) {
        int at = base + HEADER + e * ROW;
        if (obs[at + E_KIND] != 1 || obs[at + E_SIDE] != 0) {
          continue;
        }
        int id = obs[at + E_ID];
        if (id >= health[seat].length) {
          int size = Math.max(id + 1, health[seat].length * 2);
          health[seat] = Arrays.copyOf(health[seat], size);
          rate[seat] = Arrays.copyOf(rate[seat], size);
          int old = stamp[seat].length;
          stamp[seat] = Arrays.copyOf(stamp[seat], size);
          Arrays.fill(stamp[seat], old, size, -1);
        }
        float now = Math.max(0, obs[at + E_HP]) + Math.max(0, obs[at + E_SHIELD]);
        if (stamp[seat][id] < 0) {
          // First sight: what one point of its health is worth, nothing lost yet.
          float value = 0f;
          if ((obs[at + E_FLAGS] & (F_BUILDING | F_CLONE)) == 0) {
            int row = obs[at + E_ROW];
            for (int b = 0; b < bodyCount[seat]; b++) {
              if (bodyRow[seat][b] == row) {
                value = bodyValue[seat][b];
                break;
              }
            }
          }
          rate[seat][id] = value / Math.max(1f, now);
        } else if (now < health[seat][id]) {
          trade[1 - seat] += (health[seat][id] - now) * rate[seat][id];
        }
        health[seat][id] = now;
        stamp[seat][id] = stampNow;
      }
      // A troop that is gone lost what it had left.
      for (int k = 0; k < activeCount[seat]; k++) {
        int id = active[seat][k];
        if (stamp[seat][id] != stampNow && health[seat][id] > 0) {
          trade[1 - seat] += health[seat][id] * rate[seat][id];
          health[seat][id] = 0f;
        }
      }
      int live = 0;
      for (int e = 0; e < n; e++) {
        int at = base + HEADER + e * ROW;
        if (obs[at + E_KIND] == 1 && obs[at + E_SIDE] == 0) {
          active[seat][live++] = obs[at + E_ID];
        }
      }
      activeCount[seat] = live;
    }

    private void push(int seat, int seq, float own, int card, float lvl, boolean shown, float x, float y, float tick) {
      float[] rows = journal[seat];
      if (journalSize[seat] == PLAY_ROWS) {
        System.arraycopy(rows, PLAY_F, rows, 0, (PLAY_ROWS - 1) * PLAY_F);
        System.arraycopy(journalSeq[seat], 1, journalSeq[seat], 0, PLAY_ROWS - 1);
        journalSize[seat]--;
      }
      int at = journalSize[seat] * PLAY_F;
      rows[at] = own;
      rows[at + 1] = card;
      rows[at + 2] = lvl;
      rows[at + 3] = shown ? 1f : 0f;
      rows[at + 4] = shown ? x : 0f;
      rows[at + 5] = shown ? y : 0f;
      rows[at + 6] = tick;
      journalSeq[seat][journalSize[seat]++] = seq;
    }

    private void ownPlay(int seat, int action, int tick) {
      int slot = (action - 1) / Schema.TILES;
      int cell = (action - 1) % Schema.TILES;
      int card = handShown[seat][slot];
      lastGiven[seat][0] = card;
      lastGiven[seat][1] = action;
      float lvl = level;
      for (int i = 0; i < 8; i++) {
        if (deckTokens[seat * 8 + i] == card) {
          lastPlay[seat][i] = plays[seat];
          lvl = header(seat, H_DECK_LEVEL + i);
          break;
        }
      }
      plays[seat]++;
      push(seat, -1, 1f, card, lvl, true, (cell % Schema.COLS + 0.5f) / Schema.COLS,
          (cell / Schema.COLS + 0.5f) / Schema.ROWS, tick);
    }

    /** Enters and refreshes the enemy's plays as this seat's screen shows them. */
    private void enemyPlays(int seat) {
      int total = header(seat, H_ENEMY_PLAY_COUNT);
      int tick = header(seat, H_TICK);
      int shownRows = Math.min(total, ENEMY_PLAY_ROWS);
      for (int r = shownRows - 1; r >= 0; r--) {
        int seq = total - 1 - r;
        int at = seat * SEAT + H_ENEMY_PLAYS + r * ENEMY_PLAY_INTS;
        int cardId = obs[at];
        int card = at(cardOfVocab, cardId);
        int flags = obs[at + 7];
        boolean point = (flags & 2) != 0 && obs[at + 4] >= 0;
        float x = obs[at + 4] / (float) Schema.WIDTH_UNITS;
        float y = obs[at + 5] / (float) Schema.HEIGHT_UNITS;
        int ran = tick - obs[at + 3];
        if (seq >= enemySeen[seat]) {
          attributed[seat][seq & 63] = false;
          push(seat, seq, 0f, card, 0f, point, x, y, ran);
        } else {
          for (int j = journalSize[seat] - 1; j >= 0; j--) {
            if (journalSeq[seat][j] == seq) {
              float[] rows = journal[seat];
              rows[j * PLAY_F + 1] = card;
              rows[j * PLAY_F + 3] = point ? 1f : 0f;
              rows[j * PLAY_F + 4] = point ? x : 0f;
              rows[j * PLAY_F + 5] = point ? y : 0f;
              break;
            }
          }
        }
        if (card > 0 && !attributed[seat][seq & 63]) {
          attributed[seat][seq & 63] = true;
          seenCount[seat][card]++;
          seenLastSeq[seat][card] = Math.max(seenLastSeq[seat][card], seq);
          seenTick[seat][card] = Math.max(seenTick[seat][card], ran);
        }
      }
      enemySeen[seat] = total;
    }

    // ---- features ----

    private void featurize(float seconds) {
      for (int seat = 0; seat < 2; seat++) {
        featurize(seat, seconds);
      }
    }

    private void featurize(int seat, float seconds) {
      int row = index * 2 + seat;
      int base = seat * SEAT;
      int other = (1 - seat) * SEAT;
      int f = row * floatRow;
      int b = row * byteRow;
      int tick = obs[base + H_TICK];
      float elixir = obs[base + H_ELIXIR] / 10000f;

      // ---- legal actions and ability buttons ----
      bytes.put(b + B_LEGAL, mask, seat * CR_ACTIONS, CR_ACTIONS);
      boolean any = false;
      for (int action = 1; action < CR_ACTIONS && !any; action++) {
        any = mask[seat * CR_ACTIONS + action] != 0;
      }
      int ready = 0;
      for (int k = 0; k < ABILITY_ROWS; k++) {
        int at = base + H_ABILITY + k * 6;
        int out = f + O_ABILITY + k * ABILITY_F;
        for (int c = 0; c < ABILITY_F; c++) {
          floats.put(out + c, 0f);
        }
        int champion = obs[at];
        if (champion < 0) {
          continue;
        }
        boolean usable = mask[seat * CR_ACTIONS + Schema.FIRST_ABILITY + k] != 0;
        ready += usable ? 1 : 0;
        floats.put(out, at(unitOfVocab, champion));
        floats.put(out + 1, usable ? 1f : 0f);
        floats.put(out + 2, Math.max(0, Math.min(3, obs[at + 4])) / 3f);
        int state =
            switch (obs[at + 2]) {
              case 2 -> 3; // ready
              case 3 -> 4; // deploying
              case 7, 10 -> 6; // pending, casting
              case 8 -> 7; // cooldown
              case 9 -> 8; // not enough elixir
              default -> 5; // absent, spent, unknown
            };
        floats.put(out + state, 1f);
      }

      // ---- global ----
      floats.put(f + O_GLOBAL, (float) (tick / 20));
      floats.put(f + O_GLOBAL + 1, elixir);
      floats.put(f + O_GLOBAL + 2, ready);
      floats.put(f + O_GLOBAL + 3, 1f);

      // ---- entities and hazards ----
      int n = Math.min(obs[base + H_ENTITY_COUNT], Schema.MAX_ENTITIES);
      int units = 0;
      int hazards = 0;
      int unitOverflow = 0;
      int hazardOverflow = 0;
      int found = 0;
      int towers = 0;
      for (int k = 0; k < HAZARD_ROWS * HAZARD_F; k++) {
        floats.put(f + O_HAZARD + k, 0f);
      }
      for (int e = 0; e < n; e++) {
        int at = base + HEADER + e * ROW;
        int kind = obs[at + E_KIND];
        float x = clamp01(obs[at + E_X] / (float) Schema.WIDTH_UNITS);
        float y = clamp01(obs[at + E_Y] / (float) Schema.HEIGHT_UNITS);
        if (kind >= 3) {
          // Ranked and written once every entity is read: a tower's row may follow its shot.
          hazardAt[found] = at;
          hazardRank[found++] = kind == Schema.KIND_AREA ? 0 : at(hazardTier, obs[at + E_ROW], 2);
          continue;
        }
        if (kind == Schema.KIND_TOWER && towers < towerShot.length) {
          towerShot[towers++] = at(shotOfUnit, obs[at + E_ROW], -1);
        }
        if (units == maxEntities) {
          unitOverflow++;
          continue;
        }
        float[] out = scratch[units];
        int flags = obs[at + E_FLAGS];
        int vocab = obs[at + E_ROW];
        boolean tower = kind == 2;
        boolean alive = (flags & F_ALIVE) != 0;
        int buffs = 0;
        int stunMs = 0;
        int listed = Math.min(obs[at + E_BUFF_COUNT], 3);
        for (int k = 0; k < listed; k++) {
          int buff = obs[at + E_BUFF + k * 2];
          if (buff >= 0 && buff < buffFlags.length) {
            buffs |= buffFlags[buff];
            if ((buffFlags[buff] & BUFF_STUN) != 0) {
              stunMs = Math.max(stunMs, obs[at + E_BUFF + k * 2 + 1]);
            }
          }
        }
        int speed = obs[at + E_SPEED];
        int hitSpeed = obs[at + E_HIT_SPEED];
        int state = obs[at + E_STATE];
        int hp = obs[at + E_HP];
        int full = obs[at + E_MAX_HP];
        int shield = obs[at + E_SHIELD];
        int deploy = obs[at + E_DEPLOY_MS];
        out[0] = tower ? ((flags & F_KING) != 0 ? 1f : 2f) : ((flags & F_BUILDING) != 0 ? 4f : 3f);
        out[1] = obs[at + E_SIDE] == 0 ? 1f : 0f;
        out[2] = at(unitOfVocab, vocab);
        out[3] = obs[at + E_LEVEL];
        out[4] = x;
        out[5] = y;
        // An enemy's health reads unknown until its bar is drawn: it is still whole.
        out[6] = hp < 0 ? (alive ? 1f : 0f) : (full > 0 ? clamp01(hp / (float) full) : 0f);
        out[7] = shield <= 0 ? 0f : (full > 0 ? clamp01(shield / (float) full) : 1f);
        out[8] = tower && (flags & F_KING) == 0 ? at(towerOfVocab, vocab) : 0f;
        out[9] = 0f;
        out[10] = (flags & F_CLONE) != 0 ? 1f : 0f;
        out[11] = (buffs & BUFF_RAGE) != 0 || speed > 100 || hitSpeed > 100 ? 1f : 0f;
        out[12] = (buffs & BUFF_SLOW) != 0 || (speed > 0 && speed < 100) || (hitSpeed > 0 && hitSpeed < 100) ? 1f : 0f;
        out[13] = (buffs & BUFF_CURSE) != 0 ? 1f : 0f;
        out[14] = (buffs & BUFF_GOBLIN_CURSE) != 0 ? 1f : 0f;
        out[15] = (buffs & BUFF_VINES) != 0 ? 1f : 0f;
        out[16] = (flags & F_CHARGED) != 0 ? 1f : 0f;
        out[17] = state == 5 ? 1f : 0f;
        out[18] = state == 3 ? 1f : 0f;
        out[19] = 0f;
        out[20] = (flags & F_INVISIBLE) != 0 ? 1f : 0f;
        out[21] = (flags & F_HIDDEN) != 0 ? 1f : 0f;
        out[22] = 0f;
        out[23] = alive && hp != 0 ? 1f : 0f;
        out[24] = deploy > 0 ? 1f : 0f;
        out[25] = Math.min(1f, deploy / 3000f);
        out[26] = Math.min(1f, stunMs < 0 ? 1f : stunMs / 3000f);
        if (seconds > 0f) {
          out[27] = clamp(obs[at + E_DX] / 1000f / seconds / 3f);
          out[28] = clamp(obs[at + E_DY] / 1000f / seconds / 3f);
        } else {
          out[27] = 0f;
          out[28] = 0f;
        }
        order[units] = units;
        units++;
      }
      for (int k = 0; k < found; k++) {
        if (hazardRank[k] == 2) {
          int shot = obs[hazardAt[k] + E_ROW];
          for (int t = 0; t < towers; t++) {
            if (towerShot[t] == shot) {
              hazardRank[k] = 3;
              break;
            }
          }
        }
      }
      // A card's volley is one hazard, at the middle of what is still in flight: the projectiles
      // of one row and side that aim within the cast's spread of the first of them.
      for (int k = 0; k < found; k++) {
        hazardVolley[k] = -1;
        if (hazardRank[k] != 1) {
          continue;
        }
        int at = hazardAt[k];
        int lead = k;
        int spread = at(castSpread, obs[at + E_ROW], -1);
        for (int j = 0; j < k && spread >= 0; j++) {
          int first = hazardAt[j];
          if (hazardVolley[j] == j
              && obs[first + E_ROW] == obs[at + E_ROW]
              && obs[first + E_SIDE] == obs[at + E_SIDE]
              && Math.abs(obs[first + E_AIM_X] - obs[at + E_AIM_X]) <= spread
              && Math.abs(obs[first + E_AIM_Y] - obs[at + E_AIM_Y]) <= spread) {
            lead = j;
            break;
          }
        }
        if (lead == k) {
          hazardVolley[k] = k;
          volleyX[k] = volleyY[k] = 0;
          volleySize[k] = 0;
        } else {
          hazardRank[k] = 4;
        }
        volleyX[lead] += obs[at + E_X];
        volleyY[lead] += obs[at + E_Y];
        volleySize[lead]++;
      }
      for (int rank = 0; rank < 4; rank++) {
        for (int k = 0; k < found; k++) {
          if (hazardRank[k] != rank) {
            continue;
          }
          if (hazards == HAZARD_ROWS) {
            // Only an area or a cast projectile left out is a loss the counter reports.
            if (rank < 2) {
              hazardOverflow++;
            } else {
              shotsDropped++;
            }
            continue;
          }
          int at = hazardAt[k];
          int out = f + O_HAZARD + hazards++ * HAZARD_F;
          floats.put(out, at(hazardOfVocab, obs[at + E_ROW]));
          floats.put(out + 1, obs[at + E_SIDE] == 0 ? 1f : 0f);
          float px = obs[at + E_X];
          float py = obs[at + E_Y];
          if (hazardVolley[k] == k) {
            px = volleyX[k] / (float) volleySize[k];
            py = volleyY[k] / (float) volleySize[k];
          }
          floats.put(out + 2, clamp01(px / Schema.WIDTH_UNITS));
          floats.put(out + 3, clamp01(py / Schema.HEIGHT_UNITS));
          floats.put(out + 4, obs[at + E_RADIUS] / (float) Schema.WIDTH_UNITS);
        }
      }
      sort(units);
      int ent = f + O_ENTITY;
      for (int k = 0; k < units; k++) {
        floats.put(ent + k * ENTITY_ROW, scratch[order[k]], 0, ENTITY_ROW);
        bytes.put(b + B_ENTITY_MASK + k, (byte) 1);
      }
      // Rows this seat filled last time and not now go back to zero.
      for (int k = units; k < lastEntityCount[seat]; k++) {
        for (int c = 0; c < ENTITY_ROW; c++) {
          floats.put(ent + k * ENTITY_ROW + c, 0f);
        }
        bytes.put(b + B_ENTITY_MASK + k, (byte) 0);
      }
      lastEntityCount[seat] = units;
      for (int k = 0; k < HAZARD_ROWS; k++) {
        bytes.put(b + B_HAZARD_MASK + k, (byte) (k < hazards ? 1 : 0));
      }
      rowInts.put(row * ROW_INTS, units);
      rowInts.put(row * ROW_INTS + 1, unitOverflow + obs[base + H_ENTITY_OVERFLOW]);
      rowInts.put(row * ROW_INTS + 2, hazardOverflow);
      rowFlagBytes.put(row * ROW_FLAGS, (byte) (any ? 1 : 0));

      // ---- hand: card in the slot, card a play would show, level, cost ----
      for (int slot = 0; slot < 4; slot++) {
        int card = obs[base + H_HAND_CARD + slot];
        int token = at(cardOfVocab, card);
        boolean pending = obs[base + H_HAND_PENDING + slot] != 0;
        int out = f + O_HAND + slot * HAND_F;
        float lvl = level;
        for (int i = 0; i < 8; i++) {
          if (deckTokens[seat * 8 + i] == token) {
            lvl = obs[base + H_DECK_LEVEL + i];
          }
        }
        handShown[seat][slot] = token;
        floats.put(out, token);
        floats.put(out + 1, pending ? 0f : token);
        floats.put(out + 2, token > 0 && !pending ? lvl : 0f);
        floats.put(out + 3, token > 0 ? obs[base + H_HAND_COST + slot] : 0f);
      }

      // ---- own deck and its cycle ----
      for (int i = 0; i < DECK_ROWS; i++) {
        floats.put(f + O_DECK + i * DECK_F, deckTokens[seat * 8 + i]);
        floats.put(f + O_DECK + i * DECK_F + 1, obs[base + H_DECK_LEVEL + i]);
        int last = lastPlay[seat][i];
        int since = last < 0 ? -1 : plays[seat] - 1 - last;
        int out = f + O_DECK_X + i * DECK_X_F;
        floats.put(out, since >= 0 ? 1f : 0f);
        floats.put(out + 1, since < 0 ? 1f : Math.min(since, 8) / 8f);
        floats.put(out + 2, since >= 0 && since < QUEUE_LEN ? 1f : 0f);
      }

      // ---- recent plays, oldest first ----
      int size = journalSize[seat];
      floats.put(f + O_PLAYS, journal[seat], 0, size * PLAY_F);
      for (int k = size * PLAY_F; k < PLAY_ROWS * PLAY_F; k++) {
        floats.put(f + O_PLAYS + k, 0f);
      }
      for (int k = 0; k < PLAY_ROWS; k++) {
        bytes.put(b + B_PLAY_MASK + k, (byte) (k < size ? 1 : 0));
      }

      // ---- the enemy's cards seen so far, in the order the screen showed them ----
      int total = obs[base + H_ENEMY_PLAY_COUNT];
      int reveals = 0;
      for (int k = 0; k < REVEAL_ROWS; k++) {
        int card = obs[base + H_ENEMY_SEEN + k];
        int token = at(cardOfVocab, card);
        int out = f + O_REVEAL + k * REVEAL_F;
        int extra = f + O_REVEAL_X + k * REVEAL_X_F;
        boolean shown = card >= 0;
        if (shown) {
          reveals = k + 1;
          int last = seenLastSeq[seat][token];
          int since = last < 0 ? -1 : total - 1 - last;
          floats.put(out, token);
          floats.put(out + 1, seenTick[seat][token]);
          floats.put(out + 2, seenCount[seat][token]);
          floats.put(extra, since < 0 ? 1f : Math.min(since, 8) / 8f);
          floats.put(extra + 1, since >= QUEUE_LEN ? 1f : 0f);
        } else {
          floats.put(out, 0f);
          floats.put(out + 1, 0f);
          floats.put(out + 2, 0f);
          floats.put(extra, 0f);
          floats.put(extra + 1, 0f);
        }
      }
      for (int k = 0; k < REVEAL_ROWS; k++) {
        bytes.put(b + B_REVEAL_MASK + k, (byte) (k < reveals ? 1 : 0));
      }

      // ---- extras ----
      float counted = obs[base + H_ENEMY_ELIXIR] / 10000f;
      floats.put(f + O_EXTRAS, counted);
      floats.put(f + O_EXTRAS + 1, counted >= maxElixir - 1e-3f ? 1f : 0f);
      floats.put(f + O_EXTRAS + 2, (tick % 20) / TICKS_PER_SECOND);
      floats.put(f + O_EXTRAS + 3, Math.min(1f, obs[base + H_HAND_REFILL_MS] / 3000f));
      floats.put(f + O_EXTRAS + 4, Math.min(total, QUEUE_LEN) / (float) QUEUE_LEN);
      floats.put(f + O_EXTRAS + 5, Math.min(plays[seat], QUEUE_LEN) / (float) QUEUE_LEN);

      // ---- privileged: the other seat's hidden state, for the critic only ----
      floats.put(f + O_PRIV_GLOBAL, obs[other + H_ELIXIR] / 10000f);
      floats.put(f + O_PRIV_GLOBAL + 1, obs[other + H_HAND_REFILL_MS] / 50f);
      floats.put(f + O_PRIV_GLOBAL + 2, obs[base + H_HAND_REFILL_MS] / 50f);
      for (int slot = 0; slot < 4; slot++) {
        int card = obs[other + H_HAND_CARD + slot];
        floats.put(f + O_PRIV_HAND + slot, at(cardOfVocab, card));
      }
      for (int i = 0; i < DECK_ROWS; i++) {
        floats.put(f + O_PRIV_DECK + i, deckTokens[(1 - seat) * 8 + i]);
      }
    }

    /** Orders the unit rows by their values, so the order says nothing of creation order. */
    private void sort(int units) {
      for (int i = 1; i < units; i++) {
        int moving = order[i];
        int j = i - 1;
        while (j >= 0 && compare(scratch[order[j]], scratch[moving]) > 0) {
          order[j + 1] = order[j];
          j--;
        }
        order[j + 1] = moving;
      }
    }
  }

  private static int compare(float[] left, float[] right) {
    for (int k = 0; k < ENTITY_ROW; k++) {
      int order = Float.compare(left[k], right[k]);
      if (order != 0) {
        return order;
      }
    }
    return 0;
  }

  /** A table's entry for a vocabulary id, 0 for an id the table does not hold. */
  private static int at(int[] table, int id) {
    return at(table, id, 0);
  }

  private static int at(int[] table, int id, int missing) {
    return id < 0 || id >= table.length ? missing : table[id];
  }

  private static float clamp01(float value) {
    return value < 0f ? 0f : Math.min(value, 1f);
  }

  private static float clamp(float value) {
    return Math.max(-1f, Math.min(1f, value));
  }

  // ---------------------------------------------------------------------------------------------

  /**
   * Plays every card once, in a battle of its own against a seat that waits, and says how it went:
   * 0 not a card, 1 played without trouble, 2 the battle would not start, 3 the battle faulted,
   * 4 the play was refused as not modelled, 5 the card never became playable.
   */
  public int[] probeCards(int decisions) {
    int[] status = new int[cardRows.length];
    cursor.set(1);
    Runnable work =
        () -> {
          for (int token = cursor.getAndIncrement(); token < cardRows.length; token = cursor.getAndIncrement()) {
            status[token] = cardRows[token] == null ? 0 : probe(token, decisions);
          }
        };
    Thread[] workers = new Thread[Math.max(0, threads - 1)];
    for (int t = 0; t < workers.length; t++) {
      workers[t] = new Thread(work, "cfbot-probe");
      workers[t].start();
    }
    work.run();
    for (Thread worker : workers) {
      try {
        worker.join();
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(interrupted);
      }
    }
    return status;
  }

  private int probe(int token, int decisions) {
    if (!vocabulary.isDeckCard(cardRows[token])) {
      return 2;
    }
    int[] deck = defaultDeck.clone();
    boolean present = false;
    for (int card : deck) {
      present |= card == token;
    }
    if (!present) {
      deck[7] = token;
    }
    String[] names = new String[8];
    String[] plain = new String[8];
    for (int i = 0; i < 8; i++) {
      names[i] = cardRows[deck[i]];
      plain[i] = cardRows[defaultDeck[i]];
    }
    int[] obs = new int[2 * SEAT];
    byte[] mask = new byte[2 * CR_ACTIONS];
    int[] info = new int[Schema.INFO_INTS];
    LadderEnv env = new LadderEnv(tables, vocabulary, config);
    env.bind(IntBuffer.wrap(obs), ByteBuffer.wrap(mask), FloatBuffer.wrap(new float[2]), IntBuffer.wrap(info));
    int target = vocabulary.id(cardRows[token]);
    int played = 0;
    try {
      env.checkDeck(List.of(names), PLAIN);
      env.reset(token * 7919 + 13, List.of(names), List.of(plain), PLAIN, PLAIN);
      if (env.fault() != null) {
        return 2;
      }
      for (int step = 0; step < decisions && !env.done(); step++) {
        int action = 0;
        int cycle = 0;
        for (int slot = 0; slot < 4; slot++) {
          int open = openTile(mask, slot);
          if (open == 0) {
            continue;
          }
          if (obs[H_HAND_CARD + slot] == target) {
            action = open;
          } else if (cycle == 0) {
            cycle = open;
          }
        }
        boolean holds = false;
        for (int slot = 0; slot < 4; slot++) {
          holds |= obs[H_HAND_CARD + slot] == target;
        }
        if (action == 0 && !holds) {
          action = cycle;
        }
        boolean probing = action != 0 && holds;
        env.step(action, 0);
        if (env.fault() != null) {
          return 3;
        }
        if (info[11] != 0) {
          return 4;
        }
        if (probing && obs[H_LAST_REFUSAL] == 0) {
          played++;
        }
      }
    } catch (RuntimeException | AssertionError | StackOverflowError thrown) {
      return played == 0 && env.fault() == null ? 2 : 3;
    }
    return played > 0 ? 1 : 5;
  }

  /** An open play of a hand slot near the middle of the tiles its mask opens, or 0. */
  private static int openTile(byte[] mask, int slot) {
    int first = 1 + slot * Schema.TILES;
    int open = 0;
    for (int tile = 0; tile < Schema.TILES; tile++) {
      open += mask[first + tile] != 0 ? 1 : 0;
    }
    if (open == 0) {
      return 0;
    }
    int want = open / 2;
    for (int tile = 0; tile < Schema.TILES; tile++) {
      if (mask[first + tile] != 0 && want-- == 0) {
        return first + tile;
      }
    }
    return 0;
  }

  /**
   * The match clock, from one battle in which nobody plays: the tick double elixir starts, the
   * tick triple elixir starts, the tick overtime starts, the tick the battle ends, and the time
   * one elixir bar fills in at the start.
   */
  public int[] timeline() {
    int[] obs = new int[2 * SEAT];
    LadderEnv env = new LadderEnv(tables, vocabulary, new EnvConfig(1, level, level, 0));
    env.bind(
        IntBuffer.wrap(obs),
        ByteBuffer.wrap(new byte[2 * CR_ACTIONS]),
        FloatBuffer.wrap(new float[2]),
        IntBuffer.wrap(new int[Schema.INFO_INTS]));
    String[] names = new String[8];
    for (int i = 0; i < 8; i++) {
      names[i] = cardRows[defaultDeck[i]];
    }
    env.reset(1, List.of(names), List.of(names), PLAIN, PLAIN);
    int first = obs[H_FULL_BAR_MS];
    int[] out = {-1, -1, -1, 0, first};
    int bar = first;
    while (!env.done() && obs[H_TICK] < 40000) {
      env.step(0, 0);
      int tick = obs[H_TICK];
      if (obs[H_FULL_BAR_MS] != bar) {
        bar = obs[H_FULL_BAR_MS];
        if (out[0] < 0) {
          out[0] = tick;
        } else if (out[1] < 0) {
          out[1] = tick;
        }
      }
      if (out[2] < 0 && obs[H_OVERTIME] != 0) {
        out[2] = tick;
      }
      out[3] = tick;
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------------

  /** Static per-card and per-unit features, read through the battle core's own row readers. */
  private static final class Facts {
    static void fill(CfVector v) {
      BattleRecords records = new BattleRecords(v.tables);
      List<String> names = v.vocabulary.names();
      for (int id = 0; id < v.unitOfVocab.length; id++) {
        int token = v.unitOfVocab[id];
        if (token == 0) {
          continue;
        }
        try {
          var unit = records.unit(names.get(id));
          float[] row = new float[UNIT_F];
          stats(v, unit, row, 0);
          row[14] = unit.building() ? 1f : 0f;
          row[15] = 1f;
          System.arraycopy(row, 0, v.unitTable, token * UNIT_F, UNIT_F);
        } catch (RuntimeException unread) {
          // A row the battle core does not read keeps a zero row.
        }
      }
      for (int token = 1; token < v.cardRows.length; token++) {
        String name = v.cardRows[token];
        if (name == null || !v.vocabulary.isDeckCard(name)) {
          continue;
        }
        float[] row = new float[CARD_F];
        GameRow source = null;
        String table = null;
        for (String candidate : new String[] {"spells_characters", "spells_buildings", "spells_other"}) {
          if (v.tables.table(candidate).has(name)) {
            source = v.tables.table(candidate).row(name);
            table = candidate;
          }
        }
        int cost = source != null && source.has("ManaCost") ? source.intValue("ManaCost") : 0;
        v.cardCost[token] = cost;
        row[0] = 1f;
        row[1] = cost / 10f;
        int kind = "spells_characters".equals(table) ? 1 : "spells_buildings".equals(table) ? 2 : 3;
        try {
          var card = records.card(name);
          var unit = card.unit();
          if (unit != null) {
            int total = Math.max(1, card.total());
            row[7] = total / 16f;
            stats(v, unit, row, 8);
            v.cardChampion[token] = unit.champion();
            if (kind == 3) {
              kind = unit.building() ? 2 : 1;
            }
            boolean counts = kind == 1 && !unit.kamikaze() && unit.lifeTimeMs() <= 0;
            var second = card.secondary();
            v.cardBodies[token] =
                second != null
                    ? new int[] {v.vocabulary.id(unit.name()), v.vocabulary.id(second.name())}
                    : new int[] {v.vocabulary.id(unit.name())};
            v.cardBodyValue[token] = counts ? cost / (float) total : 0f;
          } else {
            kind = 3;
            spell(v, records, card, row);
          }
        } catch (RuntimeException unread) {
          kind = 4;
        }
        row[2] = kind == 1 ? 1f : 0f;
        row[3] = kind == 2 ? 1f : 0f;
        row[4] = kind == 3 ? 1f : 0f;
        row[5] = kind == 4 ? 1f : 0f;
        v.cardKind[token] = kind;
        System.arraycopy(row, 0, v.cardTable, token * CARD_F, CARD_F);
      }
    }

    private static float log(float value, float scale) {
      return (float) Math.log1p(Math.max(0f, value)) / scale;
    }

    private static int scaled(CfVector v, int base, org.crforge.core.pathfinding.combat.ScalingMode mode, Object rarity) {
      if (rarity == null || base <= 0) {
        return base;
      }
      var table = (org.crforge.core.pathfinding.combat.RarityTable) rarity;
      int packed = org.crforge.core.pathfinding.combat.PackedLevel.fromLevel(v.level, table);
      return org.crforge.core.pathfinding.combat.LevelScaling.scale(
          org.crforge.core.pathfinding.combat.ScalingGlobals.standard(), base, packed, mode, table);
    }

    private static void stats(CfVector v, org.crforge.core.battle.unit.UnitData unit, float[] out, int at) {
      var hp = org.crforge.core.pathfinding.combat.ScalingMode.CARD_HITPOINTS;
      var dmg = org.crforge.core.pathfinding.combat.ScalingMode.CARD_DAMAGE;
      int damage = unit.damage();
      if (damage <= 0 && unit.projectile() != null) {
        damage = unit.projectile().damage();
      }
      out[at] = log(scaled(v, unit.hitpoints(), hp, unit.rarity()), 9f);
      out[at + 1] = log(scaled(v, unit.shieldHitpoints(), hp, unit.rarity()), 9f);
      out[at + 2] = log(scaled(v, damage, dmg, unit.rarity()), 8f);
      out[at + 3] = unit.range() / 10000f;
      out[at + 4] = log(unit.speed(), 8f);
      out[at + 5] = unit.hitSpeedMs() / 3000f;
      out[at + 6] = unit.attacksAir() ? 1f : 0f;
      out[at + 7] = unit.attacksGround() ? 1f : 0f;
      out[at + 8] = unit.air() ? 1f : 0f;
      out[at + 9] = unit.targetOnlyBuildings() ? 1f : 0f;
      out[at + 10] = unit.areaDamageRadius() / 3000f;
      out[at + 11] = log(unit.mass(), 5f);
      out[at + 12] = unit.sightRange() / 10000f;
      out[at + 13] = unit.collisionRadius() / 2000f;
    }

    private static void spell(CfVector v, BattleRecords records, org.crforge.core.battle.deploy.DeployCard card, float[] row) {
      var mode = org.crforge.core.pathfinding.combat.ScalingMode.CARD_DAMAGE;
      int radius = card.radius();
      int damage = 0;
      int tower = 0;
      try {
        var globals = org.crforge.core.pathfinding.combat.ScalingGlobals.standard();
        if (card.projectile() != null) {
          var data = records.projectile(card.projectile());
          if (data.damage() <= 0 && data.spawnProjectile() != null) {
            data = records.projectile(data.spawnProjectile());
          }
          int waves = Math.max(1, card.projectileWaves());
          int packed = org.crforge.core.pathfinding.combat.PackedLevel.fromLevel(v.level, data.rarity());
          damage = (int) org.crforge.core.battle.projectile.ProjectileAmounts.damage(globals, data, packed) * waves;
          tower = (int) org.crforge.core.battle.projectile.ProjectileAmounts.towerDamage(globals, data, packed) * waves;
          radius = radius > 0 ? radius : data.radius();
        } else if (card.areaEffect() != null) {
          var data = records.areaEffect(card.areaEffect());
          int speed = data.hitSpeedMs();
          int hits = Math.max(1, speed > 0 ? data.lifeDurationMs() / speed : 1);
          damage = scaled(v, data.damage(), mode, data.rarity()) * hits;
          tower = (int) org.crforge.core.pathfinding.target.DirectHit.crownTowerDamage(data.crownTowerDamagePercent(), damage);
          radius = radius > 0 ? radius : data.radius();
        }
      } catch (RuntimeException unread) {
        // The card still plays; it only reads as dealing nothing.
      }
      row[22] = log(damage, 8f);
      row[23] = log(Math.max(0, tower), 8f);
      row[24] = radius / 5000f;
    }
  }
}
