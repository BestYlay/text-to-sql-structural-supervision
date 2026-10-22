import multiprocessing as mp
import pickle,sqlite3,time
from pathlib import Path
from collections import Counter

def _execute_worker(pipe, path, sql, max_rows, max_bytes):
    started = time.perf_counter()
    connection = None
    try:
        connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        connection.enable_load_extension(False)
        connection.execute('PRAGMA query_only=ON')
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_bytes)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, a, b, db, trigger):
            if action == sqlite3.SQLITE_FUNCTION and (b or a or '').lower() in {'load_extension', 'writefile', 'readfile'}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        connection.set_authorizer(authorize)
        cursor = connection.execute(sql)
        rows, size = [], 0
        for row in cursor:
            size += len(pickle.dumps(row, protocol=5))
            if len(rows) >= max_rows or size > max_bytes:
                pipe.send({'status': 'resource_limit', 'error': 'Result row/byte limit exceeded'})
                return
            rows.append(row)
        if len(pickle.dumps(rows, protocol=5)) > max_bytes:
            pipe.send({'status': 'resource_limit', 'error': 'Serialized result exceeds byte limit'})
            return
        pipe.send({'status': 'ok', 'rows': rows, 'seconds': time.perf_counter()-started})
    except Exception as error:
        status = 'resource_limit' if isinstance(error, sqlite3.DataError) and 'too big' in str(error).lower() else 'error'
        pipe.send({'status': status, 'error': str(error), 'type': type(error).__name__})
    finally:
        if connection is not None:
            connection.close()
        pipe.close()

def execute_readonly(path, sql, timeout=30.0, max_rows=100000, max_bytes=64*1024**2):
    ctx = mp.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_execute_worker, args=(sender, str(path), sql, max_rows, max_bytes))
    started = time.perf_counter()
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            try:
                result = receiver.recv()
            except EOFError:
                result = {'status': 'error', 'error': 'Worker exited without result'}
        else:
            result = {'status': 'timeout'}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
    result['wall_seconds'] = time.perf_counter()-started
    return result

def compare_rows(prediction, reference):
    # Mirrors the pinned official BIRD EX rule; bag is an explicit diagnostic.
    return {'ex': set(prediction) == set(reference), 'bag': Counter(prediction) == Counter(reference)}
