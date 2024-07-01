import time
import redis
import simplejson as json
import ast
import aioredis

class RedisWrapper(object):

    def __init__(self, REDIS_SETTINGS, setting_name='plan_cache'):

        DEFAULT_SETTING = REDIS_SETTINGS[setting_name]

        self.host = DEFAULT_SETTING['host']
        self.port = DEFAULT_SETTING['port']
        self.db = DEFAULT_SETTING['db']
        self.password = DEFAULT_SETTING['password']

        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)

    async def connect(self, REDIS_SETTINGS, setting_name='core_cache'):
        DEFAULT_SETTING = REDIS_SETTINGS[setting_name]
        redis_url = f"redis://:{DEFAULT_SETTING['password']}@{DEFAULT_SETTING['host']}:{DEFAULT_SETTING['port']}/{DEFAULT_SETTING['db']}"
        self.redis_client = await aioredis.create_redis_pool(redis_url, minsize=1, maxsize=20)

    def write_to_redis(self, key, value=None):
        if value is None:
            value = dict()

        value['redistime'] = int(time.time() * 1000)

        self.publish(key, value)
        self.set(key, value)

    def close(self):
        if hasattr(self, 'pubsub') and isinstance(self.pubsub, redis.client.PubSub):
            self.pubsub.close()
            del self.pubsub
        if hasattr(self, 'redis_api'):
            del self.redis_client

    def publish(self, channel, message):
        self.redis_client.publish(channel, json.dumps(message, ensure_ascii=False, ignore_nan=True))

    def subscribe(self, channel):
        p = self.redis_client.pubsub()
        p.subscribe(channel)
        return p

    def set(self, key, value, ex=None):
        self.redis_client.set(key, json.dumps(value, ensure_ascii=False, ignore_nan=True), ex=ex)

    def get(self, key, default=None):
        try:
            byte_res = self.redis_client.get(key)
            try:
                return json.loads(byte_res)
            except ValueError as e:
                return ast.literal_eval(byte_res.decode())
        except Exception as e:
            return default

    def ts_set(self, key, value):
        self.set(key, {'ts': time.time(), 'data': value})

    def ts_get(self, key):
        return self.get(key, {}).get('data')

    def get_ts(self, key):
        return self.get(key, {}).get('ts', 0)

    def set_1day(self, key, value):
        if isinstance(value, dict):
            value['redistime'] = time.time()
        self.redis_client.set(key, json.dumps(value, ensure_ascii=False, ignore_nan=True), ex=3600 * 24)

    def keys(self, pattern='*'):
        lst = self.redis_client.keys(pattern=pattern)
        for i, d in enumerate(lst):
            lst[i] = d.decode()
        return lst

    def get_redis_depth(self, exchange, symbol):
        return self.get(f'{exchange}_{symbol}_depth')

    def list_push(self, name, *values, side, ex=None):
        lst = [json.dumps(i, ensure_ascii=False, ignore_nan=True) for i in values]
        if side == 'r':
            self.redis_client.rpush(name, *lst)
        elif side == 'l':
            self.redis_client.lpush(name, *lst)
        if ex:
            self.redis_client.expire(name, ex)

    def list_pop(self, name, side, count=1):
        if side == 'r':
            byte_res = self.redis_client.rpop(name, count)
        elif side == 'l':
            byte_res = self.redis_client.lpop(name, count)
        else:
            return []

        try:
            try:
                return [json.loads(x) for x in byte_res]
            except ValueError as e:
                return [ast.literal_eval(x) for x in byte_res]
        except Exception as e:
            return []

    def ltrim(self, name, start=0, end=-1):
        self.redis_client.ltrim(name, start, end)

    def lrange(self, name, start=0, end=-1):
        byte_res = self.redis_client.lrange(name, start, end)
        result = [json.loads(i) for i in byte_res]
        return result

    def list_move(self, name, target_name, src="LEFT", dest="RIGHT"):  # default left pop add to right
        self.redis_client.lmove(name, target_name, src, dest)

    def delete(self, key):
        return self.redis_client.delete(key)

    def flush_all(self):
        return self.redis_client.flushall()


if __name__ == '__main__':
    redis_cli = RedisWrapper('p_lock')